import os
from pathlib import Path
import pandas as pd
import numpy as np
from battimal.data.battery_data import EchemMeasurement, CellData, BatteryDataset
from scipy.interpolate import make_splrep
import warnings


class RPT_ISUILCC_LFPGr(EchemMeasurement):
    """
    Class for keeping track of data from an electrochemical measurement.
    'IVSt' refers the the key variables in electrochemical data: current (I), voltage (V), step number (S), and 
    time (t). Optionally, other outputs from a battery test can be included, like power, charge throughput, ...
    This class also requires the type of measurement being stored, like 'formation' or 'RPT'. It is intended that 
    for any specific measurement type, there will be a subclass that inherits this to define specific attributes 
    and methods, such as unique plotting, or parsers to read the data in from a raw data file. Parsers could also
    be a defined function that outputs several EchemMeasurement objects, one for each measurement in the raw file.

    For ISU-ILCC-LFPGr, add capacity, SOC variables.
    Create extractors for different attributes. All extractors simply take in 'self' and return something else.
    """
    def __init__(self, measurement_type: str, this_rpt: pd.DataFrame):
        # Assert type is RPT
        assert measurement_type in ['RPT']

        extractors = {
            'charge_capacity': get_charge_capacity,
            # 'charge_energy': get_charge_energy,
            'charge_cv_time': get_charge_cv_time,
            'discharge_capacity_C/3': lambda rpt, rate='C/3': get_discharge_capacity(rpt, rate),
            'discharge_capacity_1C': lambda rpt, rate='1C': get_discharge_capacity(rpt, rate),
            'discharge_energy_C/3': lambda rpt, rate='C/3': get_discharge_energy(rpt, rate),
            'discharge_energy_1C': lambda rpt, rate='1C': get_discharge_energy(rpt, rate),
            'coulombic_efficiency': get_coulombic_efficiency,
            'coulombic_inefficiency': lambda rpt: 1 - get_coulombic_efficiency(rpt),
            'log_coulombic_inefficiency': lambda rpt: np.log(1 - get_coulombic_efficiency(rpt)),
            'energy_efficiency': get_energy_efficiency,
            'energy_inefficiency': lambda rpt: 1 - get_energy_efficiency(rpt),
            'log_energy_inefficiency': lambda rpt: np.log(1 - get_energy_efficiency(rpt)),
            'discharge_QV_dQdV_dVdQ': get_discharge_QV_dQdV_dVdQ,   # 1000-point C/3 spline and derivative curves
            'DeltaV_SOC': get_DeltaV_SOC,                           # Splines of 1C charge and discharge V(SOC) difference
        }
        # Extractors for QV features at various voltages
        V_interpolation = np.linspace(2.0, 3.5, 1000)
        discharge_QV_feature_voltages = [2.25, 3, 3.2, 3.28, 3.35]
        for voltage in discharge_QV_feature_voltages:
            index = np.argmin(np.abs(V_interpolation - voltage))
            extractors[f'discharge_QV_{int(voltage*1000)}mV'] = lambda rpt, index=index: rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index]['Capacity_Ah']
        # Extractors for dQdV features at various voltages
        discharge_dQdV_feature_voltages = [3.14, 3.16, 3.17, 3.235, 3.26, 3.2625, 3.265, 3.29, 3.305]
        for voltage in discharge_dQdV_feature_voltages:
            index = np.argmin(np.abs(V_interpolation - voltage))
            extractors[f'discharge_dQdV_{int(voltage*1000)}mV'] = lambda rpt, index=index: rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index]['dQdV_Ah/V']
        # Extractors for dVdQ features at various SOCs
        soc_interpolation = np.linspace(0, 1, 1000)
        discharge_dVdQ_feature_SOCs = [0.03, 0.10, 0.25, 0.50, 0.70, 0.775, 0.8, 0.825, 0.90]
        for soc in discharge_dVdQ_feature_SOCs:
            index = np.argmin(np.abs(soc_interpolation - soc))
            extractors[f'discharge_dVdQ_{int(soc*100)}SOC'] = lambda rpt, index=index: rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index]['dVdQ_V/Ah']
        # Extractors for DeltaV(SOC) features at various SOCs
        DeltaV_SOC_feature_SOCs = [0.03, 0.05, 0.10, 0.40, 0.60, 0.93]
        for soc in DeltaV_SOC_feature_SOCs:
            index = np.argmin(np.abs(soc_interpolation - soc))
            extractors[f'DeltaV_{int(soc*100)}SOC'] = lambda rpt, index=index: rpt.extracts['DeltaV_SOC'].iloc[index]['DeltaV']

        # Init the EchemMeasurement object and add ISU-ILCC specific attributes
        super().__init__(measurement_type,
                         Current_A = this_rpt['Current (A)'].to_list(),
                         Voltage_V = this_rpt['Voltage (V)'].to_list(),
                         Time_s = this_rpt['Time (s)'].to_list(),
                         Step = this_rpt['Step Number'].to_list(),
                         extractors=extractors,
                        )
        self.Measurement_Time_s = this_rpt['Measurement Time (s)'].to_list()
        self.Capacity_Ah = this_rpt['Capacity (Ah)'].to_list()
        self.Step_Description = this_rpt['Step Description'].to_list()
        self.State = this_rpt['State'].to_list()
        self.SOC = this_rpt['SOC'].to_list()

    def to_df(self) -> pd.DataFrame:
        """
        Convert the RPT_ISUILCC_LFPGr data to a pandas DataFrame.
        :return: A pandas DataFrame with the measurement data.
        """
        return pd.DataFrame({
            'Current_A': self.Current_A,
            'Voltage_V': self.Voltage_V,
            'Step': self.Step,
            'Time_s': self.Time_s,
            'Measurement_Time_s': self.Measurement_Time_s,
            'Capacity_Ah': self.Capacity_Ah,
            'Step_Description': self.Step_Description,
            'State': self.State,
            'SOC': self.SOC
        })
    
    @staticmethod
    def from_h5(measurement_type: str, hdf5: pd.HDFStore, measurement_key: str, ignore_measurement_data: bool = False):
        """
        Load the EchemMeasurement data from an HDF5 file.
        :param measurement_type: The type of measurement being loaded.
        :param hdf5: The HDFStore object to load the data from.
        :param measurement_key: The key for the measurement data in the HDF5 file.
        :param ignore_measurement_data: If True, ignore the actual measurement data and return an empty EchemMeasurement object.
        :return: An EchemMeasurement object with the loaded data.
        """
        if not ignore_measurement_data:
            this_rpt = hdf5[measurement_key]
            this_rpt = this_rpt.rename(columns={
                'Current_A': 'Current (A)',
                'Voltage_V': 'Voltage (V)',
                'Step': 'Step Number',
                'Time_s': 'Time (s)',
                'Measurement_Time_s': 'Measurement Time (s)',
                'Capacity_Ah': 'Capacity (Ah)',
                'Step_Description': 'Step Description',
                'State': 'State',
                'SOC': 'SOC'
            })
        else:
            this_rpt = {
                'Current (A)':[],
                'Voltage (V)':[],
                'Step':[],
                'Time (s)':[],
                'Capacity (Ah)':[], 
                'Measurement Time (s)':[], 
                'Step Description':[], 
                'State':[], 
                'SOC':[]
            }
            this_rpt = pd.DataFrame(this_rpt)
        return RPT_ISUILCC_LFPGr(measurement_type, this_rpt)    

class CellData_ISUILCC_LFPGr(CellData):
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
        # ISU-ILCC NMC-Gr specific metadata
        # DOD, Cdis, Cchg by cell group

        # ISU-ILCC measurement extractors
        measurement_extractors = {}
        for rate in ['C/3', '1C']:
            measurement_extractors[f'relative_discharge_capacity_{rate}'] = lambda cell_data, rate=rate: get_relative_discharge_capacity(cell_data, rate)
            measurement_extractors[f'log_relative_discharge_capacity_{rate}'] = lambda cell_data, rate=rate, transform='log': get_relative_discharge_capacity(cell_data, rate, transform)
            measurement_extractors[f'relative_discharge_energy_{rate}'] = lambda cell_data, rate=rate: get_relative_discharge_energy(cell_data, rate)
            measurement_extractors[f'log_relative_discharge_energy_{rate}'] = lambda cell_data, rate=rate, transform='log': get_relative_discharge_energy(cell_data, rate, transform)
            
        measurement_extractors[f'discharge_delta_QV_dQdV'] = lambda cell_data: get_delta_QV_dQdV_table(cell_data)
        measurement_extractors[f'delta_DeltaV_SOC'] = lambda cell_data: get_delta_DeltaV_SOC_table(cell_data)

        # Deltas, log(Deltas) of all QV, dQdV, and DeltaV(SOC) features    
        discharge_QV_feature_voltages = discharge_QV_feature_voltages = [2.25, 3, 3.2, 3.28, 3.35]
        for voltage in discharge_QV_feature_voltages:
            measurement_extractors[f'discharge_delta_QV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage: get_delta_QV_feature(cell_data, voltage)
            measurement_extractors[f'discharge_log_abs_delta_QV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage, transform='log_abs': get_delta_QV_feature(cell_data, voltage, transform)
            measurement_extractors[f'discharge_relative_QV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage: get_relative_QV_feature(cell_data, voltage)
            measurement_extractors[f'discharge_log_relative_QV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage, transform='log': get_relative_QV_feature(cell_data, voltage, transform)
        
        discharge_dQdV_feature_voltages = [3.14, 3.16, 3.17, 3.235, 3.26, 3.2625, 3.265, 3.29, 3.305]
        for voltage in discharge_dQdV_feature_voltages:
            measurement_extractors[f'discharge_delta_dQdV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage: get_delta_dQdV_feature(cell_data, voltage)
            measurement_extractors[f'discharge_log_abs_delta_dQdV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage, transform='log_abs': get_delta_dQdV_feature(cell_data, voltage, transform)
            measurement_extractors[f'discharge_relative_dQdV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage: get_relative_dQdV_feature(cell_data, voltage)
            measurement_extractors[f'discharge_log_relative_dQdV_{int(voltage*1000)}mV'] = lambda cell_data, voltage=voltage, transform='log': get_relative_dQdV_feature(cell_data, voltage, transform)
        
        discharge_dVdQ_feature_SOCs = [0.03, 0.10, 0.25, 0.50, 0.70, 0.775, 0.8, 0.825, 0.90]
        for soc in discharge_dVdQ_feature_SOCs:
            measurement_extractors[f'discharge_delta_dVdQ_{int(soc*100)}SOC'] = lambda cell_data, soc=soc: get_delta_dVdQ_feature(cell_data, soc)
            measurement_extractors[f'discharge_log_abs_delta_dVdQ_{int(soc*100)}SOC'] = lambda cell_data, soc=soc, transform='log_abs': get_delta_dVdQ_feature(cell_data, soc, transform)
            measurement_extractors[f'discharge_relative_dVdQ_{int(soc*100)}SOC'] = lambda cell_data, soc=soc: get_relative_dVdQ_feature(cell_data, soc)
            measurement_extractors[f'discharge_log_relative_dVdQ_{int(soc*100)}SOC'] = lambda cell_data, soc=soc, transform='log': get_relative_dVdQ_feature(cell_data, soc, transform)
        
        DeltaV_SOC_feature_SOCs = [0.03, 0.05, 0.10, 0.40, 0.60, 0.93]
        for soc in DeltaV_SOC_feature_SOCs:     
            measurement_extractors[f'delta_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, soc=soc: get_delta_DeltaV_SOC_feature(cell_data, soc)
            measurement_extractors[f'log_abs_delta_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, soc=soc, transform='log_abs': get_delta_DeltaV_SOC_feature(cell_data, soc, transform)
            measurement_extractors[f'relative_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, soc=soc: get_relative_DeltaV_SOC_feature(cell_data, soc)
            measurement_extractors[f'log_relative_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, soc=soc, transform='log': get_relative_DeltaV_SOC_feature(cell_data, soc, transform)

        for stat in ['var', 'mean', 'log_var', 'log_abs_mean']:
            for var in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V', 'Delta_dVdQ_V/Ah']:
                measurement_extractors[f'{stat}_{var}'] = lambda cell_data, var=var, stat=stat: get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data, var, stat)
            measurement_extractors[f'{stat}_DeltaV_SOC'] = lambda cell_data, stat=stat: get_delta_DeltaV_SOC_statistic(cell_data, stat)

        measurement_extractors['remaining_days_to_80percent_C/3_capacity'] = get_rul
        measurement_extractors['log_remaining_days_to_80percent_C/3_capacity'] = lambda cell_data, transform='log': get_rul(cell_data, transform)

        # ISU-ILCC cell-level extractors
        cell_extractors = {
            'days_to_80percent_C/3_capacity': lambda cell_data: get_days_to_failure(cell_data, 0.8),
            'log_days_to_80percent_C/3_capacity': lambda cell_data: np.log(get_days_to_failure(cell_data, 0.8)),
        }

        super().__init__(cell_id, 
                         cell_metadata=cell_metadata, 
                         characterization_data=characterization_data, 
                         measurement_extractors=measurement_extractors, 
                         cell_extractors=cell_extractors)
    
    def append(self, data: EchemMeasurement, cycles: int = 0):
        return super().append(data, 
                              start_time_s=data.Time_s[0], 
                              cycles=cycles,
                              )
    
    # @staticmethod
    # def from_h5(cell_id: str, hdf5: pd.HDFStore, prefix: str = "", is_close_hdfstore: bool = True):
    #     return CellData.from_h5(cell_id, hdf5, prefix, is_close_hdfstore,
    #                            measurement_classes={'RPT_ISUILCC_LFPGr': RPT_ISUILCC_LFPGr})

class Dataset_ISUILCC_LFPGr(BatteryDataset):
    """
    ISU ILCC LFP-Gr Preprocessor
    """

    def __init__(self, 
                 raw_path: Path = Path('data_raw'),
                 is_short: bool = False, # If True, only load a subset of the cells for testing purposes
                 battery_data: list = None,
                 metadata: dict = None):
        """
        data_fpath: path to the raw data files
        output_dir: directory to save outputs from 'get_...' methods used to output data to models
        """

        if battery_data is not None:
            # Init BatteryDataset 
            super().__init__(battery_data, metadata=metadata)
            return

        # Ignore all RuntimeWarnings (repeated divide by zero warnings which we can ignore)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        dataset = []
        # Load the raw data files and cell metadata
        metadata_spreadsheet = pd.read_csv(raw_path.joinpath('CyclingParameters.csv'))
        metadata_spreadsheet = metadata_spreadsheet.rename(columns={'Group Number': 'GroupNumber', 
                                                                    'Chg C-Rate 1': 'C_chg',
                                                                    'DChg C-Rate 1': 'C_dis',
                                                                    'DoD 1': 'DoD'})
        cell_ids = metadata_spreadsheet['Cell ID']
        if is_short:
            cell_ids = cell_ids[cell_ids.isin([2, 6, 64, 65])]
        data_path = raw_path.joinpath('rpt_data/')
        for i, cell_id in enumerate(cell_ids):
            print(f'Loading data for cell {cell_id} of 65...')
            rpts_raw = load_cell_rpts(cell_id, data_path)
            if rpts_raw is None:
                continue
            metadata_this_cell = metadata_spreadsheet.to_dict(orient='records')[i]
            cell_data = CellData_ISUILCC_LFPGr(cell_id=f"cell_{cell_id}", cell_metadata=metadata_this_cell)
            for i_rpt in rpts_raw['RPT Number'].unique():
                this_rpt = rpts_raw[rpts_raw['RPT Number'] == i_rpt].copy().reset_index(drop=True)
                if i_rpt == 0:
                    t_0 = np.datetime64(this_rpt['Date (yyyy.mm.dd hh.mm.ss)'].iloc[-1])
                this_rpt = calc_soc_and_capacity(this_rpt)
                this_rpt['Measurement Time (s)'] = this_rpt['Time (s)'].to_numpy()
                this_rpt['Time (s)'] = this_rpt['Time (s)'] + (np.datetime64(this_rpt['Date (yyyy.mm.dd hh.mm.ss)'].iloc[-1]) - t_0)/np.timedelta64(1, 's')
                cell_data.append(RPT_ISUILCC_LFPGr('RPT', this_rpt), cycles=this_rpt['Num Cycles'][0])
            cell_data.run_extractors()
            dataset.append(cell_data)

        # Init BatteryDataset 
        super().__init__(dataset, metadata={
            'name': 'ISU_ILCC_LFPGr',
            'description': 'ISU ILCC LFP-Gr Dataset',
            'target_var': 'log_days_to_80percent_C/3_capacity',
            'decision_variables_univariate': 'Stress_avg',
            'decision_variables_multivariate': ['mean_C_chg', 'mean_C_dis', 'DoD'],
            'trajectory_variable': 'repeat_num',
            'splitting_test_description':  "By 'GroupNumber'",
            'splitting_train_description': "By 'GroupNumber'",
            'splitting_val_description':   "By 'GroupNumber'",
            })
    
    @staticmethod
    def from_h5(hdf5: pd.HDFStore, ignore_measurement_data: bool = False, ignore_high_dim_measurement_extracts: bool = False):
        return BatteryDataset.from_h5(Dataset_ISUILCC_LFPGr, 
                                      hdf5,
                                      celldata_classes={'CellData_ISUILCC_LFPGr': CellData_ISUILCC_LFPGr},
                                      measurement_classes={'RPT_ISUILCC_LFPGr': RPT_ISUILCC_LFPGr},
                                      ignore_measurement_data=ignore_measurement_data,
                                      ignore_high_dim_measurement_extracts=ignore_high_dim_measurement_extracts
                                      )
    
    def get_failure_matrices(self, target_var_name='log_days_to_80percent_C/3_capacity', 
                             nan_filter_method='drop_column', n_skip_measurements=1, force_consistent_vars=True):
        cell_ids, xy_pairs = super().get_failure_matrices(target_var_name, nan_filter_method=nan_filter_method, 
                                                          n_skip_measurements=n_skip_measurements, force_consistent_vars=force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            columns = x.columns
            for col in columns:
                if 'remaining_days_to_80percent_C/3_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs

    def get_rul_matrices(self, target_var_name='log_remaining_days_to_80percent_C/3_capacity', nan_filter_method='drop_column', n_skip_measurements=1):
        cell_ids, (x, y) = super().get_rul_matrices(target_var_name, None, nan_filter_method, n_skip_measurements=n_skip_measurements)
        for col in x.columns:
            if 'remaining_days_to_80percent_C/3_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_lookahead_matrices(self, target_var_name='relative_discharge_capacity_C/3',
                               nan_filter_method='drop_column', n_skip_measurements=1, force_consistent_vars=True):
        cell_ids, xy_pairs = super().get_lookahead_matrices(target_var_name, nan_filter_method=nan_filter_method,
                                                            n_skip_measurements=n_skip_measurements, force_consistent_vars=force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_days_to_80percent_C/3_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs
    
    def get_autoregressive_matrices(self, target_var_name='relative_discharge_capacity_C/3', n_lags=1, is_diff_x=True, is_diff_y=True, 
                                    nan_filter_method='drop_column', n_skip_measurements=1):
        cell_ids, (x, y) = super().get_autoregressive_matrices(target_var_name=target_var_name, n_lags=n_lags, 
                                                               is_diff_x=is_diff_x, is_diff_y=is_diff_y, 
                                                               nan_filter_method=nan_filter_method, 
                                                               n_skip_measurements=n_skip_measurements)
        # drop remaining useful life target and any lags of it to prevent data leakage
        for col in x.columns:
            if 'remaining_days_to_80percent_C/3_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_trajectory_matrices(self, target_var_name='relative_discharge_capacity_C/3', chronological_var_name='repeat_num'):
        return super().get_trajectory_matrices(target_var_name, chronological_var_name)


"""
ISU-ILCC LFP-Gr data processing functions
"""
def get_preprocessed_data_files(dir_preprocessed_data:Path, cell_id:int, data_type:str = 'rpt'):
    """Returns a list of Path objects to all files containing data for this cell
       From https://github.com/REIL-UConn/rapid-soh-estimation-from-short-pulses/blob/main/rapid_soh_estimation/rapid_soh_estimation/common_methods.py
       Author Ben Nowacki

    Args:
        dir_preprocessed_data (Path): location of downloaded preprocessed data
        data_type (str): {'rpt', 'cycling'}. Wether to look for RPT or Cycling data
        cell_id (int): The cell id to find data for

    Returns:
        list: list of Path objects
    """
    assert data_type in ['rpt', 'cycling']

    dir_data = dir_preprocessed_data.joinpath(f'{data_type}_data')
    all_files = list(dir_data.glob(f'{data_type}_cell_{cell_id:02d}*'))

    def _file_part_num(file_path:Path):
        file_str = str(file_path.name)
        return int(file_str[file_str.rindex('_part') + len('_part') : file_str.rindex(f'{file_path.suffix}')])
    
    if len(all_files) == 0 or '_part' not in str(all_files[0]): 
        return all_files
    else:
        return sorted(all_files, key=_file_part_num)

def load_preprocessed_data(file_paths) -> pd.DataFrame:
    """Loads the processed data contained at the provided file path(s). Use 'get_preprocessed_data_files()' to get all file paths
       From https://github.com/REIL-UConn/rapid-soh-estimation-from-short-pulses/blob/main/rapid_soh_estimation/rapid_soh_estimation/common_methods.py
       Author Ben Nowacki

    Args:
        file_paths (Path or list): Path or list of Path objects. 

    Returns:
        pd.DataFrame: A dataframe containing the data at the provided filepath. If multiple file paths are provided, the data will be concatenated into a single dataframe
    """

    all_data = []
    if len(file_paths) == 0:
        print("WARNING: The provided list of filepaths is empty. Returning None")
        return None
    for file_path in file_paths:
        all_data.append( pd.read_csv(file_path) )
    return pd.concat(all_data, ignore_index=True)

step_descriptions = {
    1: 'Initial 1C discharge',
    2: 'Charge 1 1C 0-20% SOC',
    3: 'Charge 1 rest 20% SOC',
    4: 'Charge 1 charge pulse 0.2C 20% SOC',
    5: 'Charge 1 charge pulse 1C 20% SOC',
    6: 'Charge 1 charge pulse rest 20% SOC',
    7: 'Charge 1 discharge pulse 0.2C 20% SOC',
    8: 'Charge 1 discharge pulse 1C 20% SOC',
    9: 'Charge 1 discharge pulse rest 20% SOC',
    10: 'Charge 1 1C 20-50% SOC',
    11: 'Charge 1 rest 50% SOC',
    12: 'Charge 1 charge pulse 0.2C 50% SOC',
    13: 'Charge 1 charge pulse 1C 50% SOC',
    14: 'Charge 1 charge pulse rest 50% SOC',
    15: 'Charge 1 discharge pulse 0.2C 50% SOC',
    16: 'Charge 1 discharge pulse 1C 50% SOC',
    17: 'Charge 1 discharge pulse rest 50% SOC',
    18: 'Charge 1 1C 50-90% SOC',
    19: 'Charge 1 rest 90% SOC',
    20: 'Charge 1 charge pulse 0.2C 90% SOC',
    21: 'Charge 1 charge pulse 1C 90% SOC',
    22: 'Charge 1 charge pulse rest 90% SOC',
    23: 'Charge 1 discharge pulse 0.2C 90% SOC',
    24: 'Charge 1 discharge pulse 1C 90% SOC',
    25: 'Charge 1 discharge pulse rest 90% SOC',
    26: 'Charge 1 1C 90-100% SOC with CV finish',
    27: 'Charge 1 rest 100% SOC',
    28: 'Discharge 1 C/3 100-0% SOC',
    29: 'Discharge 1 rest 0% SOC',
    30: 'Charge 2 1C 0-90% SOC',
    31: 'Charge 2 rest 90% SOC',
    32: 'Charge 2 charge pulse 0.2C 90% SOC',
    33: 'Charge 2 charge pulse 1C 90% SOC',
    34: 'Charge 2 charge pulse rest 90% SOC',
    35: 'Charge 2 discharge pulse 0.2C 90% SOC',
    36: 'Charge 2 discharge pulse 1C 90% SOC',
    37: 'Charge 2 discharge pulse rest 90% SOC',
    38: 'Charge 2 1C 90-100% SOC with CV finish',
    39: 'Charge 2 rest 100% SOC',
    40: 'Discharge 2 1C 100-90% SOC',
    41: 'Discharge 2 rest 90% SOC',
    42: 'Discharge 2 charge pulse 0.2C 90% SOC',
    43: 'Discharge 2 charge pulse 1C 90% SOC',
    44: 'Discharge 2 charge pulse rest 90% SOC',
    45: 'Discharge 2 discharge pulse 0.2C 90% SOC',
    46: 'Discharge 2 discharge pulse 1C 90% SOC',
    47: 'Discharge 2 discharge pulse rest 90% SOC',
    48: 'Discharge 2 1C 90-0% SOC',
    49: 'Discharge 2 rest 0% SOC',
    50: 'Charge 3 1C 0-90% SOC',
    51: 'Charge 3 charge pulse 1.5C 90% SOC',
    52: 'Charge 3 charge pulse 1C 90% SOC',
    53: 'Charge 3 discharge pulse 1.5C 90% SOC',
    54: 'Charge 3 1C 90-100% SOC with CV finish',
    55: 'Charge 3 rest 100% SOC',
    56: 'Discharge 3 1C 100-90% SOC',
    57: 'Discharge 3 discharge pulse 1.5C 90% SOC',
    58: 'Discharge 3 discharge pulse 1C 90% SOC',
    59: 'Discharge 3 charge pulse 1.5C 90% SOC',
    60: 'Discharge 3 1C 90-100% SOC',
    61: 'Discharge 3 rest 100% SOC',
}
    
def load_cell_rpts(cell, data_path):
    # Load all rpts from a cell into a dataframe
    files = get_preprocessed_data_files(dir_preprocessed_data=data_path, cell_id=cell)
    if len(files) == 0:
        print(f"WARNING: No files found for cell {cell}. Returning None")
        return None
    cell_raw_df = load_preprocessed_data(files)
    cell_raw_df['Step Description'] = cell_raw_df['Step Number'].map(step_descriptions)
    return cell_raw_df

def calc_soc_and_capacity(rpt):
    for sub_cycle in ['Charge 1', 'Discharge 1', 'Charge 2', 'Discharge 2', 'Charge 3', 'Discharge 3']:
        mask = rpt['Step Description'].str.contains(sub_cycle)
        sub_rpt_df = rpt[mask]
        q = np.diff(sub_rpt_df['Capacity (Ah)'], prepend=0)
        q[q<0] = 0 # remove drops when step changes
        q[sub_rpt_df['Current (A)'] < 0] *= -1
        q = np.cumsum(q)
        if 'Discharge' in sub_cycle:
            q = np.abs(q)
        soc = q / np.max(q)
        if 'Discharge' in sub_cycle:
            soc = 1 - soc
        rpt.loc[mask, 'Capacity (Ah)'] = q
        rpt.loc[mask, 'SOC'] = soc
    return rpt

def find_cv_start_index(rpt):
    cccv_step = np.array(rpt.Step) == 26
    current = np.array(rpt.Current_A)[cccv_step]
    voltage = np.array(rpt.Voltage_V)[cccv_step]
    idx = np.argwhere(~np.isclose(current, current[0], atol=1.5e-3, rtol=0))[0][0] 
    if voltage[idx] < 3.59:
        scalar = 1.5
        atol = 1.5e-3 * scalar
        max_iters = 6
        iter = 0
        while voltage[idx] < 3.59:
            # print("increasing tolerance....")
            idx = np.argwhere(~np.isclose(current, current[0], atol=atol, rtol=0))[0][0] 
            iter += 1
            atol = atol * scalar
            if iter > max_iters:
                break
    return idx

"""
Measurement extractors
"""
def get_charge_capacity(rpt: RPT_ISUILCC_LFPGr) -> float:
    return np.max(np.array(rpt.Capacity_Ah)[np.array(rpt.Step) == 54])

def get_charge_energy(rpt: RPT_ISUILCC_LFPGr) -> float:
    mask_charge_3 = ['Charge 3' in description for description in np.array(rpt.Step_Description)]
    voltage = np.array(rpt.Voltage_V)[mask_charge_3]
    current = np.array(rpt.Current_A)[mask_charge_3]
    time = np.array(rpt.Measurement_Time_s)[mask_charge_3]
    time = time - time[0]
    power = voltage * current
    energy = np.trapz(power, time) / 3600
    return np.max(energy)

def get_charge_cv_time(rpt: RPT_ISUILCC_LFPGr) -> float:
    cccv_step = np.array(rpt.Step) == 26
    time = np.array(rpt.Measurement_Time_s)[cccv_step]
    idx = find_cv_start_index(rpt)
    return time[-1] - time[idx]

def get_discharge_capacity(rpt: RPT_ISUILCC_LFPGr, rate: str) -> float:
    if rate == 'C/3':
        return np.max(np.array(rpt.Capacity_Ah)[np.array(rpt.Step) == 28])
    elif rate == '1C':
        return np.max(np.array(rpt.Capacity_Ah)[np.array(rpt.Step) == 60])

def get_discharge_energy(rpt: RPT_ISUILCC_LFPGr, rate: str) -> float:
    if rate == 'C/3':
        mask_discharge = ['Discharge 1' in description for description in np.array(rpt.Step_Description)]
    elif rate == '1C':
        mask_discharge = ['Discharge 3' in description for description in np.array(rpt.Step_Description)]
    voltage = np.array(rpt.Voltage_V)[mask_discharge]
    current = np.array(rpt.Current_A)[mask_discharge]
    time = np.array(rpt.Measurement_Time_s)[mask_discharge]
    time = time - time[0]
    power = voltage * current
    energy = np.trapz(power, time) / 3600
    return np.max(energy)

def get_coulombic_efficiency(rpt: RPT_ISUILCC_LFPGr) -> float:
    return get_discharge_capacity(rpt, '1C') / get_charge_capacity(rpt)

def get_energy_efficiency(rpt: RPT_ISUILCC_LFPGr) -> float:
    return get_discharge_energy(rpt, '1C') / get_charge_energy(rpt)

def get_discharge_QV_dQdV_dVdQ(rpt: RPT_ISUILCC_LFPGr) -> pd.DataFrame:
    # Q(V) spline for CC discharge
    V_interpolation = np.linspace(2.0, 3.5, 1000)
    mask_discharge_1 = ['Discharge 1 C/3' in description for description in np.array(rpt.Step_Description)]
    Q_raw = np.array(rpt.Capacity_Ah)[mask_discharge_1]
    V_raw = np.array(rpt.Voltage_V)[mask_discharge_1]
    if V_raw[0] < V_interpolation[-1]:
        V_raw = np.concatenate(([V_interpolation[-1]], V_raw))
        Q_raw = np.concatenate(([0], Q_raw))
    idx_sort = np.argsort(V_raw)
    V_raw, Q_raw = V_raw[idx_sort], Q_raw[idx_sort]
    if len(np.unique(V_raw)) < len(V_raw):
        V_raw, idx_unique = np.unique(V_raw, return_index=True)
        Q_raw = Q_raw[idx_unique]
    spline = make_splrep(V_raw, Q_raw, s=5e-4)
    spline_derivative = spline.derivative()

    qv_new = spline(V_interpolation)
    dqdv_new = spline_derivative(V_interpolation)
    dvdq_new = 1 / dqdv_new
    # QV measurements don't have prior voltage point included and can start at unexpected values
    # due to high initial overpotential, causing weird spline behavior in extrapolation
    if np.any(qv_new < 0):
        mask_fix = qv_new < 0
        idx_fix = np.max(np.argwhere(mask_fix))
        qv_new[idx_fix:] = 0
        dqdv_new[idx_fix:] = 0
        dvdq_new[idx_fix:] = np.nan
    
    QV_spline = {
        'Voltage_V': V_interpolation,
        'Capacity_Ah': qv_new,
        'SOC': qv_new / np.max(qv_new),
        'dQdV_Ah/V': dqdv_new,
        'dVdQ_V/Ah': dvdq_new,
    }
    return pd.DataFrame.from_dict(QV_spline)

def get_DeltaV_SOC(rpt: RPT_ISUILCC_LFPGr) -> pd.DataFrame:
    soc_interpolate = np.linspace(0, 1, 1000)
    mask_discharge_3 = ['Discharge 3' in description and not 'rest ' in description for description in np.array(rpt.Step_Description)]
    mask_charge_3 = ['Charge 3' in description and not 'rest ' in description for description in np.array(rpt.Step_Description)]
    soc_chg = np.array(rpt.SOC)[mask_charge_3]
    V_chg = np.array(rpt.Voltage_V)[mask_charge_3]
    soc_chg, idx_sort = np.unique(soc_chg, return_index=True)
    # soc_chg = soc_chg/np.max(soc_chg)  # Normalize SOC to [0, 1]
    V_chg = V_chg[idx_sort]
    spline_chg = make_splrep(soc_chg, V_chg, s=9e-2)
    # discharge
    soc_dis = np.array(rpt.SOC)[mask_discharge_3]
    V_dis = np.array(rpt.Voltage_V)[mask_discharge_3]
    soc_dis, idx_sort = np.unique(soc_dis, return_index=True)
    V_dis = V_dis[idx_sort]
    spline_dis = make_splrep(soc_dis, V_dis, s=9e-2)
    # Delta V
    DeltaV_SOC = {
        'SOC': soc_interpolate,
        'DeltaV': spline_chg(soc_interpolate) - spline_dis(soc_interpolate),
    }
    return pd.DataFrame.from_dict(DeltaV_SOC)

"""
Cell-level extractors
"""
def get_days_to_failure(cell_data: CellData_ISUILCC_LFPGr, threshold: float) -> float:
    """
    Get the number of days until the cell reaches a certain threshold of capacity.
    """
    relative_discharge_capacity = np.array(cell_data.measurement_extracts['relative_discharge_capacity_C/3'])
    rpt_day = np.array(cell_data.start_time_day)
    # Find the index of the first value below the threshold
    index = np.argmax(relative_discharge_capacity < threshold)
    if index is None:
        ValueError("Couldn't find days to failure!")
    else:
        return rpt_day[index]

"""
Measurement extractors at cell-level (requires differences or comparisons between measurements)
"""
def get_rul(cell_data: CellData_ISUILCC_LFPGr, transform: str = None) -> CellData_ISUILCC_LFPGr:
    """
    Get remaining weeks (RPT repeats) until the cell hits the 80 percent relative capacity
    end of life threshold at C/3 rate
    """
    
    relative_discharge_capacity = np.array(cell_data.measurement_extracts['relative_discharge_capacity_C/3'])
    days = np.array(cell_data.start_time_day)
    days_to_failure = get_days_to_failure(cell_data, 0.8)
    remaining_days = days_to_failure - days
    remaining_days[remaining_days < 0] = np.nan
    remaining_days[np.isnan(relative_discharge_capacity)] = np.nan
    if transform == 'log':
        remaining_days[remaining_days <= 0] = np.nan
        remaining_days = np.log(remaining_days)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return remaining_days


def get_relative_discharge_capacity(cell_data: CellData_ISUILCC_LFPGr, rate: str, transform: str = None) -> list:
    """
    Get the relative discharge capacity for a given rate.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_capacity_{rate}']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_relative_discharge_energy(cell_data: CellData_ISUILCC_LFPGr, rate: str, transform: str = None) -> list:
    """
    Get the relative discharge energy for a given rate.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_energy_{rate}']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_delta_QV_dQdV_table(cell_data: CellData_ISUILCC_LFPGr) -> list:
    """
    Get the delta-QV and delta-dQdV curves from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts['discharge_QV_dQdV_dVdQ']
    extract_values = [None] * len(extract_tables)
    for index, QV_dQdV_weekN in enumerate(extract_tables):
        if cell_data.repeat_num[index] == 1:
            QV_dQdV_week0 = QV_dQdV_weekN.copy()
        else:
            extract_values[index] = pd.DataFrame.from_dict({
                'Voltage_V': QV_dQdV_week0['Voltage_V'],
                'SOC': QV_dQdV_week0['SOC'],
                'Delta_QV_Ah': QV_dQdV_weekN['Capacity_Ah'] - QV_dQdV_week0['Capacity_Ah'],
                'Delta_dQdV_Ah/V': QV_dQdV_weekN['dQdV_Ah/V'] - QV_dQdV_week0['dQdV_Ah/V'],
                'Delta_dVdQ_V/Ah': QV_dQdV_weekN['dVdQ_V/Ah'] - QV_dQdV_week0['dVdQ_V/Ah'],
                })
    return extract_values

def get_delta_QV_feature(cell_data: CellData_ISUILCC_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the delta-QV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_QV_{int(voltage*1000)}mV']
    extract_values = extract_values - extract_values[0]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_delta_dQdV_feature(cell_data: CellData_ISUILCC_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the delta-dQdV feature at a specific voltage from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_dQdV_{int(voltage*1000)}mV']
    extract_values = extract_values - extract_values[0]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_delta_dVdQ_feature(cell_data: CellData_ISUILCC_LFPGr, soc: float, transform: str = None) -> list:
    """
    Get the delta-dVdQ feature at a specific soc from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_dVdQ_{int(soc*100)}SOC']
    extract_values = extract_values - extract_values[0]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_relative_QV_feature(cell_data: CellData_ISUILCC_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the relative-QV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_QV_{int(voltage*1000)}mV']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values.astype(float)

def get_relative_dQdV_feature(cell_data: CellData_ISUILCC_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the relative-dQdV feature at a specific voltage from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_dQdV_{int(voltage*1000)}mV']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values.astype(float)

def get_relative_dVdQ_feature(cell_data: CellData_ISUILCC_LFPGr, soc: float, transform: str = None) -> list:
    """
    Get the relative-dVdQ feature at a specific soc from week N to week 0 for a given rate and voltage.
    """
    extract_values = cell_data.measurement_extracts[f'discharge_dVdQ_{int(soc*100)}SOC']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values.astype(float)

def get_delta_DeltaV_SOC_table(cell_data: CellData_ISUILCC_LFPGr) -> list:
    """
    Get the delta-DeltaV(SOC) curves from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts['DeltaV_SOC']
    extract_values = [None] * len(extract_tables)
    for index, DeltaV_SOC_weekN in enumerate(extract_tables):
        if cell_data.repeat_num[index] == 1:
            DeltaV_SOC_week0 = DeltaV_SOC_weekN.copy()
        else:
            extract_values[index] = pd.DataFrame.from_dict({
                'SOC': DeltaV_SOC_week0['SOC'],
                'delta_DeltaV': DeltaV_SOC_weekN['DeltaV'] - DeltaV_SOC_week0['DeltaV'],
            })
    return extract_values

def get_delta_DeltaV_SOC_feature(cell_data: CellData_ISUILCC_LFPGr, soc: float, transform: str = None) -> list:
    """
    Get the delta DeltaV_SOC feature at a specific SOC from week N to week 0.
    """
    extract_values = cell_data.measurement_extracts[f'DeltaV_{int(soc*100)}SOC']
    extract_values = extract_values - extract_values[0]
    if transform == 'log_abs':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values

def get_relative_DeltaV_SOC_feature(cell_data: CellData_ISUILCC_LFPGr, soc: float, transform: str = None) -> list:
    """
    Get the relative DeltaV_SOC feature at a specific SOC from week N to week 0.
    """
    extract_values = cell_data.measurement_extracts[f'DeltaV_{int(soc*100)}SOC']
    extract_values = extract_values / extract_values[0]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data: CellData_ISUILCC_LFPGr, var: str, statistic: str) -> list:
    """
    Get a statistic of the delta-QV curve from week N to week 0.
    """
    if var not in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V', 'Delta_dVdQ_V/Ah']:
        raise ValueError(f"Variable '{var}' is not supported. Supported variables are 'Delta_QV_Ah', 'Delta_dQdV_Ah/V', 'Delta_dVdQ_V/Ah'.")
    
    extract_tables = cell_data.measurement_extracts['discharge_delta_QV_dQdV']
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, Delta_QV_dQdV_dVdQ in enumerate(extract_tables):
        if cell_data.repeat_num[index] > 1:
            if statistic == 'var':
                stat_value = np.var(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'mean':
                stat_value = np.mean(Delta_QV_dQdV_dVdQ[var])
            elif statistic == 'log_var':
                stat_value = np.log(np.var(Delta_QV_dQdV_dVdQ[var]))
            elif statistic == 'log_abs_mean':
                stat_value = np.log(np.abs(np.mean(Delta_QV_dQdV_dVdQ[var])))
            else:
                raise ValueError(f"Statistic '{statistic}' is not supported.")
            extract_values[index] = stat_value
    return extract_values

def get_delta_DeltaV_SOC_statistic(cell_data: CellData_ISUILCC_LFPGr, statistic: str) -> list:
    """
    Get a statistic of the delta-DeltaV_SOC curve from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts['delta_DeltaV_SOC']
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, delta_DeltaV_SOC in enumerate(extract_tables):
        if cell_data.repeat_num[index] > 1:
            if statistic == 'var':
                stat_value = np.var(delta_DeltaV_SOC['delta_DeltaV'])
            elif statistic == 'mean':
                stat_value = np.mean(delta_DeltaV_SOC['delta_DeltaV'])
            elif statistic == 'log_var':
                stat_value = np.log(np.var(delta_DeltaV_SOC['delta_DeltaV']))
            elif statistic == 'log_abs_mean':
                stat_value = np.log(np.abs(np.mean(delta_DeltaV_SOC['delta_DeltaV'])))
            else:
                raise ValueError(f"Statistic '{statistic}' is not supported.")
            extract_values[index] = stat_value
    return extract_values