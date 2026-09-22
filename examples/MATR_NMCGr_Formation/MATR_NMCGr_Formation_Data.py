import os
from pathlib import Path
import pandas as pd
import numpy as np
from battimal.data.battery_data import EchemMeasurement, CellData, BatteryDataset
from scipy.interpolate import make_splrep
from scipy.stats import skew, kurtosis
import warnings
import h5py

"""
Notes:
- try d^2q/dv^2 features (for this data set and others) using the spline fit?
"""

class RPT_MATR_NMCGr(EchemMeasurement):
    """
    Class for keeping track of data from an electrochemical measurement.
    'IVSt' refers the the key variables in electrochemical data: current (I), voltage (V), step number (S), and 
    time (t). Optionally, other outputs from a battery test can be included, like power, charge throughput, ...
    This class also requires the type of measurement being stored, like 'formation' or 'RPT'. It is intended that 
    for any specific measurement type, there will be a subclass that inherits this to define specific attributes 
    and methods, such as unique plotting, or parsers to read the data in from a raw data file. Parsers could also
    be a defined function that outputs several EchemMeasurement objects, one for each measurement in the raw file.

    Create extractors for different attributes. All extractors simply take in 'self' and return something else.
    """
    def __init__(self, measurement_type: str,
                 current: np.ndarray,
                 voltage: np.ndarray,
                 time: np.ndarray,
                 measurement_time: np.ndarray,
                 step: np.ndarray,
                 discharge_capacity: np.ndarray,
                 discharge_differential_capacity: np.ndarray,
                 charge_capacity: np.ndarray,
                 charge_differential_capacity: np.ndarray,
                 ):
        # Assert type is RPT
        assert measurement_type in ['RPT']

        extractors = {
            'discharge_Cb20_QV_dQdV': get_discharge_Cb20_QV_dQdV,
            'discharge_Cb5_QV_dQdV': get_discharge_Cb5_QV_dQdV,
            'charge_Cb5_QV_dQdV': get_charge_Cb5_QV_dQdV,
            'DeltaV_SOC_Cb5': get_DeltaV_SOC_Cb5,
        }

        # Extractors for QV and dQdV features at various voltages
        voltages = [3.46, 3.5, 3.56, 3.585, 3.63, 3.7, 3.82, 3.9, 4.3]
        for v in voltages:
            for feature in ['Capacity_Ah']: #, 'dQdV_Ah/V' ### dQdV extractors not working right now
                extractors[f'discharge_Cb20_{feature}_{int(v*1000)}mV'] = lambda rpt, voltage=v, feature=feature, direction='discharge', rate='C/20': get_QV_dQdV_value(rpt, voltage, feature, direction, rate)
                extractors[f'charge_Cb5_{feature}_{int(v*1000)}mV'] = lambda rpt, voltage=v, feature=feature, direction='charge', rate='C/5': get_QV_dQdV_value(rpt, voltage, feature, direction, rate)
                extractors[f'discharge_Cb5_{feature}_{int(v*1000)}mV'] = lambda rpt, voltage=v, feature=feature, direction='discharge', rate='C/5': get_QV_dQdV_value(rpt, voltage, feature, direction, rate)

        socs = [0.05, 0.08, 0.1, 0.2, 0.8]
        for soc in socs:
            extractors[f'DeltaV_SOC_Cb5_{int(soc*100)}'] = lambda rpt, soc=soc: get_DeltaV_SOC_value(rpt, soc)
       
        # Init the EchemMeasurement object and add ISU-ILCC specific attributes
        super().__init__(measurement_type,
                         Current_A = current.tolist(),
                         Voltage_V = voltage.tolist(),
                         Time_s = time.tolist(),
                         Step = step.tolist(),
                         extractors=extractors,
                        )
        self.Measurement_Time_s = measurement_time.tolist()
        self.Discharge_Capacity_Ah = discharge_capacity.tolist()
        self.Discharge_DifferentialCapacity_Ah_V = discharge_differential_capacity.tolist()
        self.Charge_Capacity_Ah = charge_capacity.tolist()
        self.Charge_DifferentialCapacity_Ah_V = charge_differential_capacity.tolist()

    def to_df(self) -> pd.DataFrame:
        """
        Convert the RPT_MATR_NMCGr data to a pandas DataFrame.
        :return: A pandas DataFrame with the measurement data.
        """
        return pd.DataFrame({
            'Current_A': self.Current_A,
            'Voltage_V': self.Voltage_V,
            'Step': self.Step,
            'Time_s': self.Time_s,
            'Measurement_Time_s': self.Measurement_Time_s,
            'Discharge_Capacity_Ah': self.Discharge_Capacity_Ah,
            'Discharge_DifferentialCapacity_Ah_V': self.Discharge_DifferentialCapacity_Ah_V,
            'Charge_Capacity_Ah': self.Charge_Capacity_Ah,
            'Charge_DifferentialCapacity_Ah_V': self.Charge_DifferentialCapacity_Ah_V,
        })
    
    @staticmethod
    def from_h5(measurement_type: str, hdf5: pd.HDFStore, measurement_key: str):
        """
        Load the EchemMeasurement data from an HDF5 file.
        :param measurement_type: The type of measurement being loaded.
        :param hdf5: The HDFStore object to load the data from.
        :param measurement_key: The key for the measurement data in the HDF5 file.
        :return: An EchemMeasurement object with the loaded data.
        """
        measurement_data = hdf5[measurement_key]
        return RPT_MATR_NMCGr(measurement_type,
                            current = measurement_data['Current_A'].to_numpy(),
                            voltage = measurement_data['Voltage_V'].to_numpy(),
                            time = measurement_data['Time_s'].to_numpy(),
                            measurement_time = measurement_data['Measurement_Time_s'].to_numpy(),
                            step = measurement_data['Step'].to_numpy(),
                            discharge_capacity = measurement_data['Discharge_Capacity_Ah'].to_numpy(),
                            discharge_differential_capacity = measurement_data['Discharge_DifferentialCapacity_Ah_V'].to_numpy(),
                            charge_capacity = measurement_data['Charge_Capacity_Ah'].to_numpy(),
                            charge_differential_capacity = measurement_data['Charge_DifferentialCapacity_Ah_V'].to_numpy(),
        )
    

class CellData_MATR_NMCGr(CellData):
    """
    Class for keeping track of data from electrochemical measurements for an individual battery cell.
    If more chronological tracking states are desired, a new class can be created that inherits from this class.

    Each type of data/project/modeling effort has its own needs for extracting data from a specific set of tests.
    The class will have a dict of extractors, where each entry is the name of an extracted quantity and a function  
    that does the extracting from the cell testing data. These are assumed to be run in order if certain extracts
    rely on one another. Calculated values are stored in the extracts dict with the same name. A simple example
    of this might be the cycles until some end-of-life threshold. A more complex example would be the delta-Q(V)
    features used in early life prediction models, which require taking the difference of extracted Q(V) curves
    from two different measurements.
    """
    def __init__(self, cell_id, cell_metadata: dict = None, characterization_data: dict = None):
        """
        Initialize the CellData object with a cell ID.

        :param cell_metadata: cell metadata, which is the cell ID and any other input metadata
        :param characterization_data: characterization data, which is a dictionary of non-electrochemical characterization data from the cell
        :param measurements: list of EchemMeasurement objects for this cell
        :param measurement_type: type of measurement being stored, like 'formation' or 'RPT'
        :param start_time_s: start time of each measurement in seconds since beginning of test
        :param cycle: cycle number of each measurement
        :param repeat_num: repetition number of each measurement according to its type
        :param measurement_extractors: dict of functions for extracting quantities from the measurements. This is intended
            for calculating quantities that require comparison between multiple measurements, for example, capacity loss since
            beginning of life, or delta-Q(V) features. Extracted quantities are stored in the extracts dict in each 
            EchemMeasurement object in the list of measurements. If extracted quantities are specific to a certain type of
            EchemMeasurement, the extractor function should handle that. These functions should always input the CellData object
            and the name of the extract to be calculated, and return the CellData object, modifying the extracts in each
            measurement but not changing the length of measurements.
        :param extractors: dict of functions for extracting quantities from the CellData object.
        :param extracts: dict of extracted quantities from the CellData
        """
        measurement_extractors = {
            'discharge_Cb20_delta_QV_dQdV': lambda cell_data, direction='discharge', rate='C/20': get_delta_QV_dQdV_table(cell_data, direction=direction, rate=rate),
            'charge_Cb5_delta_QV_dQdV': lambda cell_data, direction='charge', rate='C/5': get_delta_QV_dQdV_table(cell_data, direction=direction, rate=rate),
            'discharge_Cb5_delta_QV_dQdV': lambda cell_data, direction='discharge', rate='C/5': get_delta_QV_dQdV_table(cell_data, direction=direction, rate=rate),
            'delta_DeltaV_SOC_Cb5': get_delta_DeltaV_SOC_table,
            'remaining_cycles_to_80pct_capacity': lambda cell_data, threshold=0.8: get_rul(cell_data, threshold),
            'log_remaining_cycles_to_80pct_capacity': lambda cell_data, threshold=0.8: np.log(get_rul(cell_data, threshold)),
        }
        
        voltages = [3.46, 3.5, 3.56, 3.585, 3.63, 3.7, 3.82, 3.9, 4.3]
        for v in voltages:
            for feature in ['Capacity_Ah']: # , 'dQdV_Ah/V' ### dQdV extractors not working right now
                measurement_extractors[f'discharge_Cb20_delta_{feature}_{int(v*1000)}mV'] = lambda cell_data, voltage=v, feature=feature, direction='discharge', rate='C/20': get_delta_QV_dQdV_value(cell_data, voltage, feature, direction, rate)
                measurement_extractors[f'charge_Cb5_delta_{feature}_{int(v*1000)}mV'] = lambda cell_data, voltage=v, feature=feature, direction='charge', rate='C/5': get_delta_QV_dQdV_value(cell_data, voltage, feature, direction, rate)
                measurement_extractors[f'discharge_Cb5_delta_{feature}_{int(v*1000)}mV'] = lambda cell_data, voltage=v, feature=feature, direction='discharge', rate='C/5': get_delta_QV_dQdV_value(cell_data, voltage, feature, direction, rate)
        
        socs = [0.05, 0.08, 0.1, 0.2, 0.8]
        for soc in socs:
            measurement_extractors[f'delta_DeltaV_SOC_Cb5_{int(soc*100)}'] = lambda cell_data, soc=soc: get_delta_DeltaV_SOC_value(cell_data, soc)
        
        for stat in ['var', 'mean', 'skew', 'kurtosis', 'log_var', 'log_abs_mean']:
            for var in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V']:
                measurement_extractors[f'discharge_Cb20_{stat}_{var}'] = lambda cell_data, var=var, stat=stat, direction='discharge', rate='C/20': get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data, var, stat, direction, rate)
                measurement_extractors[f'charge_Cb5_{stat}_{var}'] = lambda cell_data, var=var, stat=stat, direction='charge', rate='C/5': get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data, var, stat, direction, rate)
                measurement_extractors[f'discharge_Cb5_{stat}_{var}'] = lambda cell_data, var=var, stat=stat, direction='discharge', rate='C/5': get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data, var, stat, direction, rate)
            measurement_extractors[f'delta_DeltaV_SOC_Cb5_{stat}'] = lambda cell_data, stat=stat: get_delta_DeltaV_SOC_statistic(cell_data, stat)

        cell_extractors = {
            #### these were extracted manually by the original authors, but there is code to calculate it
            # 'cycle_life': get_cycle_life, 
            # 'log_cycle_life': lambda cell_data: np.log(get_cycle_life(cell_data)),
        }

        super().__init__(cell_id, 
                         cell_metadata=cell_metadata, 
                         characterization_data=characterization_data, 
                         measurement_extractors=measurement_extractors, 
                         cell_extractors=cell_extractors)
    
    def append(self, data: EchemMeasurement, cycles: int = 1):
        return super().append(data, 
                              start_time_s = data.Time_s[0], 
                              cycles = cycles,
                              )

class Dataset_MATR_NMCGr(BatteryDataset):
    """
    MATR NMC-Gr Formation Study Preprocessor
    """

    def __init__(self, data_fpath: Path = Path('data_raw'),
                 is_short=False):
        """
        data_fpath: path to the folder wtih the raw data folder
        output_dir: directory to save outputs from 'get_...' methods used to output data to models
        """
        # Ignore all RuntimeWarnings (repeated divide by zero warnings which we can ignore)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Load all cell metadata, raw data files, and pre-extracted values
        # Cell metadata and targets
        cell_metadata = pd.read_csv(os.path.join(data_fpath, 'Formation_2022-Parameter.csv'))
        targets = pd.read_csv(os.path.join(data_fpath, 'one_time_features_041524.csv'))
        # add electrode info to metadata for replicating plots from original work
        electrode_info = pd.read_csv(os.path.join(data_fpath, 'electrode_info_04152024.csv'))
        # only keep post-formation electrode info
        electrode_info = electrode_info[electrode_info['cycle_index'] == 0]
        

        # Merge cell metadata with targets and meta
        cell_metadata = cell_metadata.merge(targets, on='seq_num')
        cell_metadata = cell_metadata.merge(electrode_info, on='seq_num')
        cell_metadata = cell_metadata.rename(columns={
                'cell_id': 'cell_serial',
                'seq_num': 'cell_id',
        })
        cell_metadata['cycle_life'] = cell_metadata['regu_life']
        cell_metadata['log_cycle_life'] = np.log(cell_metadata['cycle_life'])
        del targets, electrode_info

        # Measurement metadata and extracted features
        measurements_summary = pd.read_csv(os.path.join(data_fpath, 'rpt_summary_041524.csv'))
        features_hppc = pd.read_csv(os.path.join(data_fpath, 'delta_features_041524.csv'))     

        # list of data files for raw data
        data_files = os.listdir(os.path.join(data_fpath, 'Aging (extracted from BEEP format)'))

        # Init dataset list and iterate through cells
        dataset = []
        if is_short:
            cell_nums = cell_metadata['cell_id'].unique()[:5]
        else:
            cell_nums = cell_metadata['cell_id'].unique()
        for cell_num in cell_nums:
            print(cell_num)
            if cell_num == 206 or cell_num == 269 or cell_num == 292:
                print(f'Skipping cell {cell_num}; missing hppc data')
                continue
            if cell_num == 115 or cell_num == 152:
                print(f'Skipping cell {cell_num}; noisy rpt data')
                continue
            # Cell metadata and targets
            cell_meta = cell_metadata[cell_metadata['cell_id'] == cell_num]
            cell_meta.pop('cell_id')
            cell_meta = cell_meta.to_dict(orient='records')[0]

            # Cell measurement extracts
            cell_measurements = measurements_summary[measurements_summary['seq_num'] == cell_num]
            # Process DCIR features from this cell
            cell_features_hppc = features_hppc[features_hppc['seq_num'] == cell_num]
            # drop end columns that we don't want: seq_num, diag_pos
            cell_features_hppc.pop('seq_num')
            cell_features_hppc.pop('diag_pos')
            # they saved the raw hppc value at cycle 0 and then all consecutive values are deltas from that, so lets add them back together
            cell_features_hppc = cell_features_hppc.reset_index(drop=True)
            for col in cell_features_hppc.columns[1:]:
                cell_features_hppc.loc[1:, col] += cell_features_hppc[col].iloc[0]
            cell_measurements = cell_measurements.merge(cell_features_hppc, on='cycle_index')
            cell_measurements['relative_discharge_capacity'] = cell_measurements['regu_cap'] / cell_measurements['regu_cap'].iloc[0]
            # Only keep RPTs from cell measurements above 60% relative discharge capacity (ignoring funky trajectories that would be hard to model)
            cell_measurements = cell_measurements[cell_measurements['relative_discharge_capacity'] >= 0.6]
            cell_measurements = cell_measurements.reset_index(drop=True)

            # Grab raw data for this cell
            cell_files = [f for f in data_files if f'{cell_num}' in f]
            # ignore aging cycles for now, only rpts
            cell_files = [f for f in cell_files if 'rpt_data' in f]
            rpt_ivt = pd.read_pickle(os.path.join(data_fpath, 'Aging (extracted from BEEP format)', cell_files[0]))

            # Init CellData object
            cell_data = CellData_MATR_NMCGr(cell_num, cell_metadata=cell_meta)

            cycle_minus1 = 0
            # ignore first rpt because it's just hppc data (no capacity cycles)
            for cycle in cell_measurements['cycle_index'].values[1:]:
                cycle_hppc = cycle
                cycle_rpt_Cb20 = cycle + 1
                cycle_rpt_Cb5 = cycle + 2
                # Index raw rpt_ivt data from all 3 cycles together
                rpt_raw = rpt_ivt[(rpt_ivt['cycle_index'] == cycle_hppc) | (rpt_ivt['cycle_index'] == cycle_rpt_Cb20) | (rpt_ivt['cycle_index'] == cycle_rpt_Cb5)]
                rpt = RPT_MATR_NMCGr(
                    measurement_type='RPT',
                    current=rpt_raw['current'].to_numpy(),
                    voltage=rpt_raw['voltage'].to_numpy(),
                    time=rpt_raw['test_time'].to_numpy(),
                    measurement_time=rpt_raw['test_time'].to_numpy() - rpt_raw['test_time'].to_numpy()[0],
                    step=rpt_raw['step_index'].to_numpy(),
                    discharge_capacity=rpt_raw['discharge_capacity'].to_numpy(),
                    discharge_differential_capacity=rpt_raw['discharge_dQdV'].to_numpy(),
                    charge_capacity=rpt_raw['charge_capacity'].to_numpy(),
                    charge_differential_capacity=rpt_raw['charge_dQdV'].to_numpy(),
                )
                # Copy all extracts from the original authors
                cycle_measurements = cell_measurements[cell_measurements['cycle_index'] == cycle]
                cycle_measurements.pop('cycle_index')
                cycle_measurements.pop('seq_num')
                cycle_measurements.pop('diag_pos')
                cycle_measurements = cycle_measurements.to_dict(orient='records')[0]
                rpt.extracts = cycle_measurements

                cell_data.append(rpt, cycles = cycle - cycle_minus1)
                cycle_minus1 = cycle
                
            cell_data.run_extractors()
            dataset.append(cell_data)
            del cell_data

        # Init BatteryDataset 
        super().__init__(dataset, metadata={
            'name': 'MATR_NMCGr_Formation',
            'description': 'MATR NMC-Gr Formation Study',
            'target_var': 'log_cycle_life',
            'decision_variables': ['formation_temperature', 'ocv_time', 'formation_charge_current_1', 'formation_cutoff_voltage_1', 'formation_charge_current_2', 'formation_verification_repeat'],
            'trajectory_variable': 'cycle',
            'splitting_test_description':  "By cell_id",
            'splitting_train_description': "By cell_id",
            'splitting_val_description':   "By cell_id",
            })
    
    @staticmethod
    def from_h5(hdf5: pd.HDFStore, ignore_measurement_data: bool = False, ignore_high_dim_measurement_extracts: bool = False):
        return BatteryDataset.from_h5(Dataset_MATR_NMCGr,
                                      hdf5,
                                      celldata_classes={'CellData_MATR_NMCGr': CellData_MATR_NMCGr},
                                      measurement_classes={'RPT_MATR_NMCGr': RPT_MATR_NMCGr},
                                      ignore_measurement_data=ignore_measurement_data,
                                      ignore_high_dim_measurement_extracts=ignore_high_dim_measurement_extracts
                                      )

    def get_failure_matrices(self, target_var_name='log_cycle_life', nan_filter_method='drop_column'):
        cell_ids, xy_pairs = super().get_failure_matrices(target_var_name=target_var_name, nan_filter_method=nan_filter_method)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_cycles_to_80pct_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs

    def get_rul_matrices(self, target_var_name='log_remaining_cycles_to_80pct_capacity', nan_filter_method='drop_row'):
        cell_ids, (x, y) = super().get_rul_matrices(target_var_name=target_var_name, nan_filter_method=nan_filter_method)
        for col in x.columns:
            if 'remaining_cycles_to_80pct_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)

    def get_lookahead_matrices(self, target_var_name='relative_discharge_capacity'):
        cell_ids, xy_pairs = super().get_lookahead_matrices(target_var_name=target_var_name)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_cycles_to_80pct_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs
    
    def get_autoregressive_matrices(self, target_var_name='relative_discharge_capacity', n_lags=1, step_size=1, is_diff_x=True, is_diff_y=True, nan_filter_method='drop_row'):
        cell_ids, (x, y) = super().get_autoregressive_matrices(target_var_name=target_var_name, 
                                                               n_lags=n_lags, step_size=step_size, 
                                                               is_diff_x=is_diff_x, is_diff_y=is_diff_y, 
                                                               nan_filter_method=nan_filter_method)
        # drop remaining useful life target and any lags of it to prevent data leakage
        for col in x.columns:
            if 'remaining_cycles_to_80pct_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_trajectory_matrices(self, target_var_name='relative_discharge_capacity', chronological_var_name='cycle'):
        return super().get_trajectory_matrices(target_var_name=target_var_name, chronological_var_name=chronological_var_name)



"""
Cell extractors for the MATR NMC-Gr Formation dataset.
"""
def get_cycle_life(cell_data: CellData_MATR_NMCGr, threshold: float = 0.8) -> float:
    """
    Get the number of cycles until the cell reaches a certain threshold of capacity.
    """
    relative_discharge_capacity = np.array(cell_data.measurement_extracts['relative_discharge_capacity'])
    cycles = np.array(cell_data.cycle)
    if np.any(relative_discharge_capacity < threshold):
        # linearly interp to find the cycle at which the threshold is reached
        idx = np.argwhere(relative_discharge_capacity < threshold)[0][0]
        cycle_life = np.interp(threshold, 
                               [relative_discharge_capacity[idx-1], relative_discharge_capacity[idx]], 
                               [cycles[idx-1], cycles[idx]])
    else:
        cycle_life = cycles[-1]
    return cycle_life

    
"""
Measurement extractors for the MATR NMC-Gr Formation dataset.
"""    
def get_discharge_Cb20_QV_dQdV(rpt: RPT_MATR_NMCGr) -> pd.DataFrame:
    """
    Extract the QV and dQdV curves for the C/20 discharge.
    """
    rpt = rpt.to_df()
    rpt = rpt[rpt['Step'] == 13]
    rpt = rpt[['Voltage_V', 'Discharge_Capacity_Ah', 'Discharge_DifferentialCapacity_Ah_V']]
    rpt = rpt.rename(columns={
        'Discharge_Capacity_Ah': 'Capacity_Ah',
        'Discharge_DifferentialCapacity_Ah_V': 'dQdV_Ah/V',
    })
    rpt = rpt.reset_index(drop=True)
    # Smooth dQdV curve with rolling mean
    rpt['dQdV_Ah/V'] = rpt['dQdV_Ah/V'].rolling(window=5, center=True, min_periods=1).mean()
    return rpt

def get_discharge_Cb5_QV_dQdV(rpt: RPT_MATR_NMCGr) -> pd.DataFrame:
    """
    Extract the QV and dQdV curves for the C/5 discharge.
    """
    rpt = rpt.to_df()
    rpt = rpt[rpt['Step'] == 16]
    rpt = rpt[['Voltage_V', 'Discharge_Capacity_Ah', 'Discharge_DifferentialCapacity_Ah_V']]
    rpt = rpt.rename(columns={
        'Discharge_Capacity_Ah': 'Capacity_Ah',
        'Discharge_DifferentialCapacity_Ah_V': 'dQdV_Ah/V',
    })
    rpt = rpt.reset_index(drop=True)
    # Smooth dQdV curve with rolling mean
    rpt['dQdV_Ah/V'] = rpt['dQdV_Ah/V'].rolling(window=5, center=True, min_periods=1).mean()
    return rpt
    
    
def get_charge_Cb5_QV_dQdV(rpt: RPT_MATR_NMCGr) -> pd.DataFrame:
    """
    Extract the QV and dQdV curves for the C/5 charge.
    """
    rpt = rpt.to_df()
    rpt = rpt[rpt['Step'] == 15]
    rpt = rpt[['Voltage_V', 'Charge_Capacity_Ah', 'Charge_DifferentialCapacity_Ah_V']]
    rpt = rpt.rename(columns={
        'Charge_Capacity_Ah': 'Capacity_Ah',
        'Charge_DifferentialCapacity_Ah_V': 'dQdV_Ah/V',
    })
    rpt = rpt.reset_index(drop=True)
    # Smooth dQdV curve with rolling mean
    rpt['dQdV_Ah/V'] = rpt['dQdV_Ah/V'].rolling(window=5, center=True, min_periods=1).mean()
    return rpt
    
    
def get_DeltaV_SOC_Cb5(rpt: RPT_MATR_NMCGr) -> pd.DataFrame:
    """
    Extract the DeltaV(SOC) curve for the C/5 cycle.
    """
    cb5_charge = get_charge_Cb5_QV_dQdV(rpt)
    cb5_charge['soc'] = cb5_charge['Capacity_Ah'] / cb5_charge['Capacity_Ah'].max()
    cb5_discharge = get_discharge_Cb5_QV_dQdV(rpt)
    cb5_discharge['soc'] = 1 - (cb5_discharge['Capacity_Ah'] / cb5_discharge['Capacity_Ah'].max())

    soc = np.linspace(0, 1, 100)
    cb5_charge_interp = np.interp(soc, cb5_charge['soc'], cb5_charge['Voltage_V'])
    cb5_discharge_interp = np.interp(soc, cb5_discharge['soc'].to_numpy()[::-1], cb5_discharge['Voltage_V'].to_numpy()[::-1])
    DeltaV = cb5_charge_interp - cb5_discharge_interp
    return pd.DataFrame.from_dict({
        'SOC': soc,
        'DeltaV': DeltaV,
    })
    

def get_QV_dQdV_value(rpt: RPT_MATR_NMCGr, voltage: float, feature: str, direction: str, rate: str) -> float:
    """
    Get the QV or dQdV value at a specific voltage from a charge or discharge curve at C/20 or C/5
    """
    assert feature in ['Capacity_Ah', 'dQdV_Ah/V'], "Feature must be 'Capacity_Ah' or 'dQdV_Ah/V'"

    if rate == 'C/20':
        assert direction == 'discharge', "Only discharge curves are available for C/20 rate"
        return float(np.interp(voltage, rpt.extracts['discharge_Cb20_QV_dQdV']['Voltage_V'], rpt.extracts['discharge_Cb20_QV_dQdV'][feature]))
    elif rate == 'C/5':
        if direction == 'charge':
            return float(np.interp(voltage, rpt.extracts['charge_Cb5_QV_dQdV']['Voltage_V'], rpt.extracts['charge_Cb5_QV_dQdV'][feature]))
        elif direction == 'discharge':
            return float(np.interp(voltage, rpt.extracts['discharge_Cb5_QV_dQdV']['Voltage_V'], rpt.extracts['discharge_Cb5_QV_dQdV'][feature]))
        else:
            raise ValueError("Direction must be 'charge' or 'discharge'")
    else:
        raise ValueError("Rate must be 'C/20' or 'C/5'")
   
def get_DeltaV_SOC_value(rpt: RPT_MATR_NMCGr, soc: float) -> float:
    """
    Get the DeltaV(SOC) value at a specific SOC from the C/5 cycle.
    """
    return float(np.interp(soc, rpt.extracts['DeltaV_SOC_Cb5']['SOC'], rpt.extracts['DeltaV_SOC_Cb5']['DeltaV']))

"""
Measurement extractors at the cell level for the MATR NMC-Gr Formation dataset.
"""
def get_rul(cell_data: CellData_MATR_NMCGr, threshold: float = 0.8) -> list:
    """
    Get the remaining useful life (RUL) at each cycle until the cell reaches a certain threshold of capacity.
    """
    cycles = np.array(cell_data.cycle)
    cycle_life = cell_data.cell_metadata['cycle_life']
    rul = cycle_life - cycles
    rul[rul < 0] = np.nan
    return rul.tolist()

def get_delta_QV_dQdV_table(cell_data: CellData_MATR_NMCGr, direction: str, rate: str) -> list:
    """
    Get the delta-QV and delta-dQdV curves from week N to week 0 for a given rate.
    """
    if rate == 'C/20':
        assert direction == 'discharge', "Only discharge curves are available for C/20 rate"
        extract_tables = cell_data.measurement_extracts['discharge_Cb20_QV_dQdV']
    elif rate == 'C/5':
        if direction == 'charge':
            extract_tables = cell_data.measurement_extracts['charge_Cb5_QV_dQdV']
        elif direction == 'discharge':
            extract_tables = cell_data.measurement_extracts['discharge_Cb5_QV_dQdV']
        else:
            raise ValueError("Direction must be 'charge' or 'discharge'")
    else:
        raise ValueError("Rate must be 'C/20' or 'C/5'")

    extract_values = [None] * len(extract_tables)
    QV_dQdV_cycle0 = None
    for index, QV_dQdV_cycleN in enumerate(extract_tables):
        if index == 0:
            QV_dQdV_cycle0 = QV_dQdV_cycleN.copy()
        elif index > 0:
            extract_values[index] = pd.DataFrame.from_dict({
                'Delta_QV_Ah': QV_dQdV_cycleN['Capacity_Ah'] - QV_dQdV_cycle0['Capacity_Ah'],
                'Delta_dQdV_Ah/V': QV_dQdV_cycleN['dQdV_Ah/V'] - QV_dQdV_cycle0['dQdV_Ah/V'],
                })
    return extract_values

def get_delta_DeltaV_SOC_table(cell_data: CellData_MATR_NMCGr) -> list:
    """
    Get the delta-DeltaV(SOC) curve from week N to week 0.
    """
    extract_tables = cell_data.measurement_extracts['DeltaV_SOC_Cb5']
    extract_values = [None] * len(extract_tables)
    DeltaV_SOC_cycle0 = None
    for index, DeltaV_SOC_cycleN in enumerate(extract_tables):
        if index == 0:
            DeltaV_SOC_cycle0 = DeltaV_SOC_cycleN.copy()
        elif index > 0:
            extract_values[index] = pd.DataFrame.from_dict({
                'delta_DeltaV_SOC': DeltaV_SOC_cycleN['DeltaV'] - DeltaV_SOC_cycle0['DeltaV'],
                'SOC': DeltaV_SOC_cycleN['SOC'],
            })
    return extract_values

def get_delta_QV_dQdV_value(cell_data: CellData_MATR_NMCGr, voltage: float, feature: str, direction: str, rate: str) -> float:
    """
    Get the delta-QV or delta-dQdV value at a specific voltage from week N to week 0 for a given rate and voltage.
    """
    assert feature in ['Capacity_Ah', 'dQdV_Ah/V'], "Feature must be 'Capacity_Ah' or 'dQdV_Ah/V'"
    if rate == 'C/20':
        rate = 'Cb20'
    elif rate == 'C/5':
        rate = 'Cb5'
    else:
        raise ValueError("Rate must be 'C/20' or 'C/5'")
    extract_values = np.array(cell_data.measurement_extracts[f'{direction}_{rate}_{feature}_{int(voltage*1000)}mV'])
    extract_values = extract_values - extract_values[0]
    return extract_values

def get_delta_DeltaV_SOC_value(cell_data: CellData_MATR_NMCGr, soc: float) -> float:
    """
    Get the delta-DeltaV(SOC) value at a specific SOC from week N to week 0.
    """
    extract_values = np.array(cell_data.measurement_extracts[f'DeltaV_SOC_Cb5_{int(soc*100)}'])
    extract_values = extract_values - extract_values[0]
    return extract_values

def get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data: CellData_MATR_NMCGr, var: str, statistic: str, direction: str, rate: str) -> list:
    """
    Get a statistic of the delta-QV curve from week N to week 0.
    """
    if var not in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V']:
        raise ValueError(f"Variable '{var}' is not supported. Supported variables are 'Delta_QV_Ah', 'Delta_dQdV_Ah/V'.")
    
    # extract_tables = cell_data.measurement_extracts['discharge_delta_QV_dQdV']
    if rate == 'C/20':
        assert direction == 'discharge', "Only discharge curves are available for C/20 rate"
        extract_tables = cell_data.measurement_extracts['discharge_Cb20_delta_QV_dQdV']
    elif rate == 'C/5':
        if direction == 'charge':
            extract_tables = cell_data.measurement_extracts['charge_Cb5_delta_QV_dQdV']
        elif direction == 'discharge':
            extract_tables = cell_data.measurement_extracts['discharge_Cb5_delta_QV_dQdV']
        else:
            raise ValueError("Direction must be 'charge' or 'discharge'")
    else:
        raise ValueError("Rate must be 'C/20' or 'C/5'")

    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, Delta_QV_dQdV_dVdQ in enumerate(extract_tables):
        if Delta_QV_dQdV_dVdQ is not None:
            if statistic == 'var':
                stat_value = np.var(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'mean':
                stat_value = np.mean(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'skew':
                stat_value = skew(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'kurtosis':
                stat_value = kurtosis(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'log_var':
                stat_value = np.log(np.var(Delta_QV_dQdV_dVdQ[var]))
            elif statistic == 'log_abs_mean':
                stat_value = np.log(np.abs(np.mean(Delta_QV_dQdV_dVdQ[var])))
            else:
                raise ValueError(f"Statistic '{statistic}' is not supported.")
            extract_values[index] = stat_value
    return extract_values

def get_delta_DeltaV_SOC_statistic(cell_data: CellData_MATR_NMCGr, statistic: str) -> list:
    """
    Get a statistic of the delta-DeltaV_SOC curve from week N to week 0.
    """
    extract_tables = cell_data.measurement_extracts['delta_DeltaV_SOC_Cb5']
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, delta_DeltaV_SOC in enumerate(extract_tables):
        if delta_DeltaV_SOC is not None:
            if statistic == 'var':
                stat_value = np.var(delta_DeltaV_SOC['delta_DeltaV_SOC'])
            elif statistic == 'mean':
                stat_value = np.mean(delta_DeltaV_SOC['delta_DeltaV_SOC'])
            elif statistic == 'skew':
                stat_value = skew(delta_DeltaV_SOC['delta_DeltaV_SOC'])
            elif statistic == 'kurtosis':
                stat_value = kurtosis(delta_DeltaV_SOC['delta_DeltaV_SOC'])
            elif statistic == 'log_var':
                stat_value = np.log(np.var(delta_DeltaV_SOC['delta_DeltaV_SOC']))
            elif statistic == 'log_abs_mean':
                stat_value = np.log(np.abs(np.mean(delta_DeltaV_SOC['delta_DeltaV_SOC'])))
            else:
                raise ValueError(f"Statistic '{statistic}' is not supported.")
            extract_values[index] = stat_value
    return extract_values