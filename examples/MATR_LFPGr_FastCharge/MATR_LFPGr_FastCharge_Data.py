import os
from pathlib import Path
import pandas as pd
import numpy as np
from battimal.data.battery_data import EchemMeasurement, CellData, BatteryDataset
from scipy.interpolate import make_splrep
from scipy.stats import skew, kurtosis
import warnings
import h5py

class Cycle_MATR_LFPGr(EchemMeasurement):
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
                 charge_capacity: np.ndarray,
                 temperature: np.ndarray):
        # Assert type is Cycle
        assert measurement_type in ['Cycle']

        extractors = {
            'charge_capacity': get_charge_capacity,
            # 'charge_energy': get_charge_energy,
            'charge_cv_time': get_charge_cv_time,
            'discharge_capacity': get_discharge_capacity,
            'discharge_energy': get_discharge_energy,
            'coulombic_efficiency': get_coulombic_efficiency,
            'coulombic_inefficiency': lambda rpt: 1 - get_coulombic_efficiency(rpt),
            'log_coulombic_efficiency': lambda rpt: np.log(get_coulombic_efficiency(rpt)),
            'energy_efficiency': get_energy_efficiency,
            'energy_inefficiency': lambda rpt: 1 - get_energy_efficiency(rpt),
            'log_energy_efficiency': lambda rpt: np.log(get_energy_efficiency(rpt)),
            'discharge_QV_dQdV_dVdQ': get_discharge_QV_dQdV_dVdQ,   # 1000-point 4C discharge spline and derivative curves
        }
        # Extractors for QV and dQdV features at various voltages
        voltages = [2.2, 2.48, 2.8, 3.0, 3.07, 3.15, 3.35]
        for v in voltages:
            for feature in ['QV', 'dQdV']:
                extractors[f'discharge_{feature}_{int(v*1000)}mV'] = lambda rpt, voltage=v, feature=feature: get_QV_dQdV_value(rpt, voltage, feature)
        # dQdV integral over voltage windows
        voltage_windows = [(3.25, 3.45), (2.8, 3.25), (3.15, 3.25), (2.8, 3.15), (2.2, 2.8)]
        for v in voltage_windows:
            extractors[f"discharge_dQdV_integral_{int(v[0]*1000)}mV_{int(v[1]*1000)}mV"] = lambda rpt, voltage_window=v: get_dQdV_integral(rpt, voltage_window)
       
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
        self.Charge_Capacity_Ah = charge_capacity.tolist()
        self.Temperature_C = temperature.tolist()

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
            'Discharge_Capacity_Ah': self.Discharge_Capacity_Ah,
            'Charge_Capacity_Ah': self.Charge_Capacity_Ah,
            'Temperature_C': self.Temperature_C
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
            measurement_data = hdf5[measurement_key]
            return Cycle_MATR_LFPGr(measurement_type,
                                current = measurement_data['Current_A'].to_numpy(),
                                voltage = measurement_data['Voltage_V'].to_numpy(),
                                time = measurement_data['Time_s'].to_numpy(),
                                measurement_time = measurement_data['Measurement_Time_s'].to_numpy(),
                                step = measurement_data['Step'].to_numpy(),
                                discharge_capacity = measurement_data['Discharge_Capacity_Ah'].to_numpy(),
                                charge_capacity = measurement_data['Charge_Capacity_Ah'].to_numpy(),
                                temperature = measurement_data['Temperature_C'].to_numpy()
            )
        else:
            return Cycle_MATR_LFPGr(measurement_type,
                                current = np.array([]),
                                voltage = np.array([]),
                                time = np.array([]),
                                measurement_time = np.array([]),
                                step = np.array([]),
                                discharge_capacity = np.array([]),
                                charge_capacity = np.array([]),
                                temperature = np.array([])
            )
    

class CellData_MATR_LFPGr(CellData):
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
        # '0' cycle for delta features
        cycle_0 = 10
        measurement_extractors = {
            'relative_discharge_capacity':get_relative_discharge_capacity,
            # 'log_relative_discharge_capacity': lambda cell_data, transform='log': get_relative_discharge_capacity(cell_data, transform=transform),
            'relative_discharge_energy': get_relative_discharge_energy,
            # 'log_relative_discharge_energy': lambda cell_data, transform='log': get_relative_discharge_energy(cell_data, transform=transform),
            'discharge_delta_QV_dQdV': lambda cell_data, cycle_0=cycle_0: get_delta_QV_dQdV_table(cell_data, cycle_0),
        }
        
        # Deltas, log(Deltas) of all QV, dQdV features    
        voltages = [2.2, 2.48, 2.8, 3.0, 3.07, 3.15, 3.35]
        for v in voltages:
            measurement_extractors[f'discharge_delta_QV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, cycle_0=cycle_0: get_delta_QV_feature(cell_data, voltage, cycle_0=cycle_0)
            measurement_extractors[f'discharge_log_abs_delta_QV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, cycle_0=cycle_0, transform='log_abs': get_delta_QV_feature(cell_data, voltage, cycle_0, transform)
            # measurement_extractors[f'discharge_relative_QV_{int(v*1000)}mV'] = lambda cell_data, voltage=v: get_relative_QV_feature(cell_data, voltage)
            # measurement_extractors[f'discharge_log_relative_QV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, transform='log': get_relative_QV_feature(cell_data, voltage, transform)
            measurement_extractors[f'discharge_delta_dQdV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, cycle_0=cycle_0: get_delta_dQdV_feature(cell_data, voltage, cycle_0)
            measurement_extractors[f'discharge_log_abs_delta_dQdV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, cycle_0=cycle_0, transform='log_abs': get_delta_dQdV_feature(cell_data, voltage, cycle_0, transform)
            # measurement_extractors[f'discharge_relative_dQdV_{int(v*1000)}mV'] = lambda cell_data, voltage=v: get_relative_dQdV_feature(cell_data, voltage)
            # measurement_extractors[f'discharge_log_relative_dQdV_{int(v*1000)}mV'] = lambda cell_data, voltage=v, transform='log': get_relative_dQdV_feature(cell_data, voltage, transform)
        # deltas, log(Deltas) of dQdV integral over voltage windows
        voltage_windows = [(3.25, 3.45), (2.8, 3.25), (3.15, 3.25), (2.8, 3.15), (2.2, 2.8)]
        for v in voltage_windows:
            measurement_extractors[f'discharge_delta_dQdV_integral_{int(v[0]*1000)}mV_{int(v[1]*1000)}mV'] = lambda cell_data, voltage_window=v, cycle_0=cycle_0: get_delta_dQdV_integral_feature(cell_data, voltage_window, cycle_0)
            measurement_extractors[f'discharge_log_abs_delta_dQdV_integral_{int(v[0]*1000)}mV_{int(v[1]*1000)}mV'] = lambda cell_data, voltage_window=v, cycle_0=cycle_0, transform='log_abs': get_delta_dQdV_integral_feature(cell_data, voltage_window, cycle_0, transform)
        
        for stat in ['var', 'mean', 'skew', 'kurtosis', 'log_var', 'log_abs_mean']:
            for var in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V']:
                measurement_extractors[f'{stat}_{var}'] = lambda cell_data, var=var, stat=stat, cycle_0=cycle_0: get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data, var, stat, cycle_0)

        measurement_extractors['remaining_cycle_life'] = get_rul
        measurement_extractors['log_remaining_cycle_life'] = lambda cell_data, transform='log': get_rul(cell_data, transform=transform)
        cell_extractors = {
            'cycle_life': get_cycle_life,
            'log_cycle_life': lambda cell_data: np.log(get_cycle_life(cell_data)),
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
                              charge_throughput_Ah = data.Charge_Capacity_Ah[-1] + data.Discharge_Capacity_Ah[-1],
                              )

class Dataset_MATR_LFPGr(BatteryDataset):
    """
    MATR LFP-Gr Fast-Charging Study Preprocessor
    """

    def __init__(self, 
                 data_fpath: Path = Path('data_raw'),
                 is_short=False,
                 only_study_1=False,
                 battery_data: list = None,
                 metadata: dict = None):
        """
        data_fpath: path to the folder wtih the raw data folder
        output_dir: directory to save outputs from 'get_...' methods used to output data to models
        """

        if battery_data is not None:
            # Init BatteryDataset 
            super().__init__(battery_data, metadata=metadata)
            return

        # Ignore all RuntimeWarnings (repeated divide by zero warnings which we can ignore)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if not only_study_1:
            all_batches = {
                'study_1': {
                    'batch_1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                    'batch_2': '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                    'batch_3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
                },
                'study_2': {
                    'batch_1': '2018-08-28_batchdata_updated_struct_errorcorrect.mat',
                    'batch_2': '2018-09-02_batchdata_updated_struct_errorcorrect.mat',
                    'batch_3': '2018-09-06_batchdata_updated_struct_errorcorrect.mat',
                    'batch_4': '2018-09-10_batchdata_updated_struct_errorcorrect.mat',
                    'batch_5': '2019-01-24_batchdata_updated_struct_errorcorrect.mat',
                }
            }
        else:
            all_batches = {
                'study_1': {
                    'batch_1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                    'batch_2': '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                    'batch_3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
                },
            }
        
        bad_cells = [
            # original authors say some study 1 batch 3 cells are 'noisy' and removed them
            'study_1_batch_3_cell_38',
            'study_1_batch_3_cell_3',
            'study_1_batch_3_cell_24',
            'study_1_batch_3_cell_33',
            'study_1_batch_3_cell_39',
            'study_1_batch_3_cell_30',
            # cells that I found where the data or splines are gnarly and I don't know what to do
            'study_1_batch_2_cell_41',
            'study_1_batch_2_cell_42',
            'study_1_batch_2_cell_43',
            'study_1_batch_2_cell_44',
            'study_1_batch_2_cell_45',
            'study_1_batch_2_cell_46',
            'study_1_batch_2_cell_47',
            'study_1_batch_2_cell_48',
        ]

        dataset = []
        cell_ids = []
        for study in all_batches:
            for batch in all_batches[study]:
                # Load the raw data files and cell metadata
                dpath = data_fpath.joinpath(all_batches[study][batch])
                with h5py.File(dpath, 'r') as f:
                    batch_data = f['batch']
                    if is_short:
                        num_cells = 2 #4
                    else:
                        num_cells = batch_data['summary'].shape[0]
                    for cell in range(num_cells):
                        # metadata 
                        cell_id = f"{study}_{batch}_cell_{cell+1}"
                        print(cell_id)
                        policy = f[batch_data['policy_readable'][cell,0]][:].tobytes()[::2].decode()
                        if study == 'study_1':
                            policy_values = extract_policy_study_1(policy)
                            policy_type = 'policy_1'
                            if np.any(np.isnan(policy_values)):
                                is_bad_policy = True
                                print(f"Skipping {cell_id} because policy {policy} is not in expected format")
                            else:
                                is_bad_policy = False
                                policy_values = convert_policy_1_to_policy_2(*policy_values)
                        else:
                            policy_values = [float(d) for d in policy.split('-')]
                            policy_type = 'policy_2'
                            is_bad_policy = False
                        if not is_bad_policy:
                            nominal_charging_time_minutes = calculate_nominal_charging_time(policy_type, policy_values)
                            cell_metadata = {
                                'cell_id': cell_id,
                                'study': study,
                                'batch': batch,
                                'policy_str': policy,
                                'policy_rate_1': policy_values[0],
                                'policy_rate_2': policy_values[1],
                                'policy_rate_3': policy_values[2],
                                'policy_rate_4': policy_values[3],
                                'nominal_charging_time_minutes': nominal_charging_time_minutes,
                            }
                            if cell_id in bad_cells:
                                print (f"Skipping {cell_id} because original authors or Paul say it is noisy")
                            else:
                                cell_ids.append(cell_id)
                                cell_data = CellData_MATR_LFPGr(cell_id, cell_metadata=cell_metadata)
                                cycles = f[batch_data['cycles'][cell,0]]
                                t_0 = 0.0
                                for i in range(cycles['I'].shape[0]):
                                    current = np.hstack((f[cycles['I'][i,0]]))
                                    voltage = np.hstack((f[cycles['V'][i,0]]))
                                    time = np.hstack((f[cycles['t'][i,0]]))
                                    Qd = np.hstack((f[cycles['Qd'][i,0]]))
                                    Qc = np.hstack((f[cycles['Qc'][i,0]]))
                                    temperature = np.hstack((f[cycles['T'][i,0]]))
                                    # if len(current) > 2 and max(time) > 10 and max(time) < 100 and Qd[0] == 0:
                                    if len(current) < 2:
                                        print(f"    Cycle {i}: short data, len(current) = {len(current)}!")
                                    elif max(time) < 10:
                                        print(f"    Cycle {i}: short time! max(time) = {max(time)} mins")
                                    elif max(time) > 100:
                                        print(f"    Cycle {i}: long time! max(time) = {max(time)} mins")
                                    elif Qd[0] > 1e-2:
                                        print(f"    Cycle {i}: Qd[0] != 0! Qd[0] = {Qd[0]}")

                                    try:
                                        idx_discharge = np.nonzero(current < -3.98)[0][0] - 1
                                        step = np.ones_like(current) 
                                        step[idx_discharge:] = 2
                                        current_dis = current[step == 2]
                                        idx_discharge_end = np.nonzero(np.diff(current_dis, prepend=0) > 0.1)[0][0]
                                        step[idx_discharge+idx_discharge_end:] = 3
                                        discharge_capacity = Qd[step >= 2]
                                        if discharge_capacity[0] > 1e-2:
                                            discharge_capacity = discharge_capacity - discharge_capacity[0]
                                        Qd[idx_discharge:] = discharge_capacity
                                    except:
                                        step = np.zeros_like(current) 
                                    
                                    time = time * 60
                                    cell_data.append(Cycle_MATR_LFPGr(
                                            measurement_type='Cycle',
                                            current=current,
                                            voltage=voltage,
                                            time=time + t_0,
                                            measurement_time=time,
                                            step=step,
                                            discharge_capacity=Qd,
                                            charge_capacity=Qc,
                                            temperature=temperature))
                                    t_0 += time[-1]

                                # Extracts calculated by original authors
                                if not cell_id == "study_2_batch_5_cell_32":
                                    cell_data.measurement_extracts['MATR_dcir'] =       np.hstack(f[batch_data['summary'][cell,0]]['IR'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_Qchg'] =       np.hstack(f[batch_data['summary'][cell,0]]['QCharge'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_Qdis'] =       np.hstack(f[batch_data['summary'][cell,0]]['QDischarge'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_Tavg'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tavg'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_Tmin'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tmin'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_Tmax'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tmax'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_chargetime'] = np.hstack(f[batch_data['summary'][cell,0]]['chargetime'][0,:]).tolist()
                                    cell_data.measurement_extracts['MATR_cycle'] =      np.hstack(f[batch_data['summary'][cell,0]]['cycle'][0,:]).tolist()
                                    cell_data.cell_extracts['MATR_cycle_life'] = np.array([f[batch_data['cycle_life'][cell,0]][0][0]]).tolist()[0]
                                else:
                                    cell_data.measurement_extracts['MATR_dcir'] =       np.hstack(f[batch_data['summary'][cell,0]]['IR'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_Qchg'] =       np.hstack(f[batch_data['summary'][cell,0]]['QCharge'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_Qdis'] =       np.hstack(f[batch_data['summary'][cell,0]]['QDischarge'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_Tavg'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tavg'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_Tmin'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tmin'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_Tmax'] =       np.hstack(f[batch_data['summary'][cell,0]]['Tmax'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_chargetime'] = np.hstack(f[batch_data['summary'][cell,0]]['chargetime'][0,:-1]).tolist()
                                    cell_data.measurement_extracts['MATR_cycle'] =      np.hstack(f[batch_data['summary'][cell,0]]['cycle'][0,:]).tolist()
                                    cell_data.cell_extracts['MATR_cycle_life'] = np.array([f[batch_data['cycle_life'][cell,0]][0][0]]).tolist()[0]
                               
                                cell_data.run_extractors()
                                # DCIR values in study 2 are just filled with 0, replace with NaN so we know to ignore it
                                if 'study_2' in cell_id:
                                    cell_data.measurement_extracts['MATR_dcir'] = np.full_like(np.array(cell_data.measurement_extracts['MATR_dcir']), np.nan).tolist()
                                dataset.append(cell_data)

        # Init BatteryDataset 
        super().__init__(dataset, metadata={
            'name': 'MATR_LFPGr_FastCharging',
            'description': 'MATR LFP-Gr Fast Charging Protocol (Severson) and optimization (Attia) Dataset',
            'target_var': 'log_cycle_life',
            'decision_variables_only-study-1': ['policy_rate_1', 'policy_rate_2', 'policy_rate_3', 'policy_rate_4'],
            'decision_variables_all-studies': ['policy_rate_1', 'policy_rate_2', 'policy_rate_3'],
            'trajectory_variable': 'cycle',
            'splitting_test_description':  "[cell for cell_id if 'study_1_batch_3' in cell]",
            'splitting_train_description': "[cell for cell_id if 'study_1_batch_1' in cell]",
            'splitting_val_description':   "[cell for cell_id if 'study_1_batch_2' in cell]",
            })
    
    @staticmethod
    def from_h5(hdf5: pd.HDFStore, ignore_measurement_data: bool = False, ignore_high_dim_measurement_extracts: bool = False):
        return BatteryDataset.from_h5(Dataset_MATR_LFPGr,
                                      hdf5,
                                      celldata_classes={'CellData_MATR_LFPGr': CellData_MATR_LFPGr},
                                      measurement_classes={'Cycle_MATR_LFPGr': Cycle_MATR_LFPGr},
                                      ignore_measurement_data=ignore_measurement_data,
                                      ignore_high_dim_measurement_extracts=ignore_high_dim_measurement_extracts
                                      )

    def get_failure_matrices(self, target_var_name='log_cycle_life', measurement_type='Cycle', alignment_var='cycle', 
                             cumulative=False, nan_filter_method='drop_column',
                             n_skip_measurements=10, force_consistent_vars=False):
        cell_ids, xy_pairs = super().get_failure_matrices(target_var_name, measurement_type, alignment_var,
                                                          cumulative, nan_filter_method,
                                                          n_skip_measurements, force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_cycle_life' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs

    def get_rul_matrices(self, target_var_name='log_remaining_cycle_life', measurement_type=None, nan_filter_method='drop_column', n_skip_measurements=10):
        cell_ids, (x, y) = super().get_rul_matrices(target_var_name, measurement_type, nan_filter_method, n_skip_measurements)
        for col in x.columns:
            if 'remaining_cycle_life' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_lookahead_matrices(self, target_var_name='log_cycle_life', measurement_type='Cycle', alignment_var='cycle', 
                             cumulative=False, nan_filter_method='drop_column',
                             n_skip_measurements=10, force_consistent_vars=False):
        cell_ids, xy_pairs = super().get_lookahead_matrices(target_var_name=target_var_name, measurement_type=measurement_type, alignment_var=alignment_var,
                                                          cumulative=cumulative, nan_filter_method=nan_filter_method,
                                                          n_skip_measurements=n_skip_measurements, force_consistent_vars=force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_cycle_life' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs
    

    # target_var_name, measurement_type=None, alignment_var=None,
    #                                 n_lags=1, step_size=1, is_diff_x=True, is_diff_y=True, 
    #                                 nan_filter_method='drop_row', n_skip_measurements=0
    
    def get_autoregressive_matrices(self, target_var_name='relative_discharge_capacity', measurement_type='Cycle', alignment_var='cycle',
                                    n_lags=[50], step_size=50, is_diff_x=True, is_diff_y=True, nan_filter_method='drop_row',
                                    n_skip_measurements=10):
        cell_ids, (x, y) = super().get_autoregressive_matrices(target_var_name=target_var_name, 
                                                               measurement_type=measurement_type, alignment_var=alignment_var,
                                                               n_lags=n_lags, step_size=step_size, 
                                                               is_diff_x=is_diff_x, is_diff_y=is_diff_y, 
                                                               nan_filter_method=nan_filter_method)
        # drop remaining useful life target and any lags of it to prevent data leakage
        for col in x.columns:
            if 'remaining_cycle_life' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    # , target_var_name, chronological_var_name,  measurement_type=None, n_skip_measurements=0
    def get_trajectory_matrices(self, target_var_name='relative_discharge_capacity', chronological_var_name='cycle',
                                measurement_type='Cycle', n_skip_measurements=0):
        return super().get_trajectory_matrices(target_var_name, chronological_var_name)
    

import regex
# From string of format '%fC(%f%%)-%fC', for example, '8C(35%)-3.6C', extract the C-rates and SOC %
def extract_policy_study_1(policy_str):
    pattern = r'(?P<rate>[-+]?\d*\.?\d+)(?P<unit>C)\((?P<soc>\d*\.?\d+)%\)-(?P<rest_rate>[-+]?\d*\.?\d+)C'
    match = regex.match(pattern, policy_str)
    if match:
        return [float(match.group('rate')), float(match.group('soc')), float(match.group('rest_rate'))]
    else:
        return [np.nan, np.nan, np.nan]

def calculate_nominal_charging_time(policy_type, policy):
    # Calculate the nominal charging time based on the policy type and policy values.
    if policy_type == 'policy_1':
        c_rate_1 = policy[0]
        soc_threshold = policy[1] / 100  # Convert percentage to fraction
        c_rate_2 = policy[2]
        nominal_charging_time = (soc_threshold / c_rate_1) + (0.8 - soc_threshold)/c_rate_2
    elif policy_type == 'policy_2':
        nominal_charging_time = (0.2 / policy[0]) + (0.2 / policy[1]) + (0.2 / policy[2]) + (0.2 / policy[3])
    return nominal_charging_time * 60

def convert_policy_1_to_policy_2(policy_1_C_rate_1, policy_1_soc_threshold, policy_1_C_rate_2, policy_2_socs=[0.0, 0.2, 0.4, 0.6, 0.8]):
    policy_1_soc_threshold = policy_1_soc_threshold/100
    effective_policy_2 = []
    for i in range(len(policy_2_socs) - 1):
        soc_start = policy_2_socs[i]
        soc_end = policy_2_socs[i + 1]
        if soc_end <= policy_1_soc_threshold:
            # Entire range is within the first C-rate
            effective_rate = policy_1_C_rate_1
        elif soc_start >= policy_1_soc_threshold:
            # Entire range is within the second C-rate
            effective_rate = policy_1_C_rate_2
        else:
            # Range spans both C-rates, calculate weighted average
            range_1 = policy_1_soc_threshold - soc_start
            range_2 = soc_end - policy_1_soc_threshold
            total_range = range_1 + range_2
            effective_rate = (range_1 * policy_1_C_rate_1 + range_2 * policy_1_C_rate_2) / total_range
        effective_policy_2.append(effective_rate)
    return effective_policy_2

def find_cv_start_index(rpt):
    idx_discharge = np.nonzero(np.array(rpt.Current_A) < -3.98)[0][0] - 1
    current = np.array(rpt.Current_A)[:idx_discharge]
    idx = np.argwhere(np.isclose(current, 1, atol=1.5e-3, rtol=0))
    if len(idx) > 0:
        idx = idx[-1][0]
        return idx
    else:
        return None

"""
Cell extractors for the MATR LFP-Gr Fast Charging dataset.
"""
def get_cycle_life(cell_data: CellData_MATR_LFPGr, threshold: float = 0.88) -> float:
    """
    Get the number of cycles until the cell reaches a certain threshold of capacity.
    study_2_batch_1-4 cells don't hit EOL, return nan
    """
    if "study_2" in cell_data.cell_metadata['cell_id'] and not "batch_5" in cell_data.cell_metadata['cell_id']:
        return np.nan
    
    discharge_capacity = np.array(cell_data.measurement_extracts['discharge_capacity'])
    cycles = np.array(cell_data.cycle)
    idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    discharge_capacity = discharge_capacity[idx_cycle_1:]
    cycles = cycles[idx_cycle_1:]
    # Find the index of the first value below the threshold
    index = np.argwhere(discharge_capacity < threshold)
    if len(index) == 0:
        return cycles[-1] + 1 # this is how original authors defined it
    else:
        return cycles[index[0][0]]
    
def get_rul(cell_data: CellData_MATR_LFPGr, transform: str = None) -> CellData_MATR_LFPGr:
    """
    Get remaining weeks (RPT repeats) until the cell hits the 80 percent relative capacity
    end of life threshold
    """ 
    
    cycles = np.array(cell_data.cycle)
    cycles_to_failure = get_cycle_life(cell_data, 0.8)
    # study_2 batches 1-4 were only cycled 100 times, not tested to failure
    if cycles_to_failure is np.nan:
        return np.full_like(cycles, np.nan)
    
    remaining_cycles = cycles_to_failure.astype(float) - cycles
    remaining_cycles[remaining_cycles < 0] = np.nan
    remaining_cycles[np.isnan(remaining_cycles)] = np.nan
    if transform == 'log':
        remaining_cycles[remaining_cycles <= 0] = np.nan
        remaining_cycles = np.log(remaining_cycles)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return remaining_cycles


"""
Measurement extractors for the MATR LFP-Gr Fast Charging dataset.
"""    
def get_charge_capacity(rpt: Cycle_MATR_LFPGr) -> float:
    q = np.max(rpt.Charge_Capacity_Ah)
    if q == 0.0:
        return np.nan
    else:
        return q

def get_charge_energy(rpt: Cycle_MATR_LFPGr) -> float:
    mask_charge = np.array(rpt.Step) == 1
    if np.any(mask_charge):
        voltage = np.array(rpt.Voltage_V)[mask_charge]
        current = np.array(rpt.Current_A)[mask_charge]
        time = np.array(rpt.Measurement_Time_s)[mask_charge]
        time = time - time[0]
        power = voltage * current
        energy = np.trapz(power, time) / 3600
        return np.max(energy)
    else:
        return np.nan

def get_charge_cv_time(rpt: Cycle_MATR_LFPGr) -> float:
    mask_charge = np.array(rpt.Step) == 1
    if np.any(mask_charge):
        time = np.array(rpt.Measurement_Time_s)[mask_charge]
        idx = find_cv_start_index(rpt)
        if idx is not None:
            return time[-1] - time[idx]
        else:
            return np.nan

def get_discharge_capacity(rpt: Cycle_MATR_LFPGr) -> float:
    q = np.max(rpt.Discharge_Capacity_Ah)
    if q == 0.0:
        return np.nan
    else:
        return q

def get_discharge_energy(rpt: Cycle_MATR_LFPGr) -> float:
    mask_discharge = np.array(rpt.Step) >= 2
    if np.any(mask_discharge):
        voltage = np.array(rpt.Voltage_V)[mask_discharge]
        current = np.abs(np.array(rpt.Current_A)[mask_discharge])
        time = np.array(rpt.Measurement_Time_s)[mask_discharge]
        time = time - time[0]
        power = voltage * current
        energy = np.trapz(power, time) / 3600
        return np.max(energy)
    else:
        return np.nan

def get_coulombic_efficiency(rpt: Cycle_MATR_LFPGr) -> float:
    return get_discharge_capacity(rpt) / get_charge_capacity(rpt)

def get_energy_efficiency(rpt: Cycle_MATR_LFPGr) -> float:
    return get_discharge_energy(rpt) / get_charge_energy(rpt)

def get_discharge_QV_dQdV_dVdQ(rpt: Cycle_MATR_LFPGr) -> pd.DataFrame:
    # Q(V) spline for CC discharge
    V_interpolation = np.linspace(2, 3.5, 1000)
    mask_discharge = np.array(rpt.Step) == 2
    if np.any(mask_discharge):
        Q_raw = np.array(rpt.Discharge_Capacity_Ah)[mask_discharge]
        V_raw = np.array(rpt.Voltage_V)[mask_discharge]
        if np.min(V_raw) > 2.1:
            print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because min(V) > 2.1 V")
            return None
        if np.max(V_raw) < 3.4:
            print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because max(V) < 3.4 V")
            return None
        
        if V_raw[0] < V_interpolation[-1]:
            V_raw = np.concatenate(([V_interpolation[-1]], V_raw))
            Q_raw = np.concatenate(([0], Q_raw))
        idx_sort = np.argsort(V_raw)
        V_raw, Q_raw = V_raw[idx_sort], Q_raw[idx_sort]
        if len(np.unique(V_raw)) < len(V_raw):
            V_raw, idx_unique = np.unique(V_raw, return_index=True)
            Q_raw = Q_raw[idx_unique]

        try:
            spline = make_splrep(V_raw, Q_raw, s=8e-5)
            spline_derivative = spline.derivative()

            qv_new = spline(V_interpolation)
            dqdv_new = spline_derivative(V_interpolation)
            dvdq_new = 1 / dqdv_new

            # QV measurements don't have prior voltage point included and can start at unexpected values
            # due to high initial overpotential, causing weird spline behavior in extrapolation
            if np.any(qv_new < 0):
                mask_fix = qv_new < 0
                idx_fix = np.nonzero(mask_fix)
                qv_new[idx_fix] = 0
                dqdv_new[idx_fix] = 0
                dvdq_new[idx_fix] = np.nan

            if np.any(dqdv_new > 1):
                print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because any(dQdV > 1 Ah) (spline fail)")
                return None
            if np.any(qv_new > 1.25):
                print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because any(QV > 1.25 Ah) (spline fail)")
                return None
            if np.any(qv_new < -0.1):
                print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because any(QV < -0.1 Ah) (spline fail)")
                return None
            if qv_new[-1] > 0.05:
                print(f"Skipping QV_dQdV for RPT start_time_s {rpt.Time_s[0]} because QV[-1] > 0.05 Ah (spline fail)")
                return None
            else:
                QV_spline = {
                    'Voltage_V': V_interpolation,
                    'Capacity_Ah': qv_new,
                    'dQdV_Ah/V': dqdv_new,
                    'dVdQ_V/Ah': dvdq_new,
                }
                return pd.DataFrame.from_dict(QV_spline)
        except:
            print(f"Spline fit failed! RPT start_time_s {rpt.Time_s[0]}")
            return None
    else:
        return None
    

def get_QV_dQdV_value(rpt: Cycle_MATR_LFPGr, voltage: float, feature: str) -> float:
    """
    Get the QV or dQdV value at a specific voltage for a given rate.
    """
    V_interpolation = np.linspace(2.0, 3.5, 1000)
    index = np.argmin(np.abs(V_interpolation - voltage))
    if rpt.extracts['discharge_QV_dQdV_dVdQ'] is not None:
        if feature == 'QV':
            return np.mean(rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index-20:index+20]['Capacity_Ah'])
        elif feature == 'dQdV':
            return np.mean(rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index-20:index+20]['dQdV_Ah/V'])
        else:
            raise ValueError("Feature must be 'QV' or 'dQdV'")
    else:
        return np.nan
    
def get_dQdV_integral(rpt: Cycle_MATR_LFPGr, voltage_window: tuple) -> float:
    """
    Get the integral of dQdV over a voltage window for a given rate.
    """
    V_interpolation = np.linspace(2.0, 3.5, 1000)
    idx_start = np.argmin(np.abs(V_interpolation - voltage_window[0]))
    idx_end = np.argmin(np.abs(V_interpolation - voltage_window[1]))
    if rpt.extracts['discharge_QV_dQdV_dVdQ'] is not None:
        dQdV_values = rpt.extracts['discharge_QV_dQdV_dVdQ']['dQdV_Ah/V'].iloc[idx_start:idx_end]
        return np.abs(np.trapz(dQdV_values, V_interpolation[idx_start:idx_end]))
    else:
        return np.nan
    
"""
Measurement extractors at the cell level for the MATR LFP-Gr Fast Charging dataset.
"""
def get_relative_discharge_capacity(cell_data: CellData_MATR_LFPGr, transform: str = None) -> list:
    """
    Get the relative discharge capacity for a given rate.
    """
    idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    extract_values = cell_data.measurement_extracts['discharge_capacity']
    extract_values = extract_values / extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_relative_discharge_energy(cell_data: CellData_MATR_LFPGr, transform: str = None) -> list:
    """
    Get the relative discharge energy for a given rate.
    """
    idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    extract_values = np.array(cell_data.measurement_extracts['discharge_energy'])
    extract_values = extract_values / extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_delta_QV_dQdV_table(cell_data: CellData_MATR_LFPGr, cycle_0: int) -> list:
    """
    Get the delta-QV and delta-dQdV curves from week N to week 0 for a given rate.
    """
    if cycle_0 is None:
        idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    else:
        idx_cycle_1 = cycle_0
    extract_tables = cell_data.measurement_extracts['discharge_QV_dQdV_dVdQ']
    extract_values = [None] * len(extract_tables)
    QV_dQdV_cycle0 = None
    for index, QV_dQdV_cycleN in enumerate(extract_tables):
        if cell_data.cycle[index] >= idx_cycle_1 and QV_dQdV_cycleN is not None and QV_dQdV_cycle0 is None:
            QV_dQdV_cycle0 = QV_dQdV_cycleN.copy()
        elif QV_dQdV_cycleN is not None and index > idx_cycle_1:
            extract_values[index] = pd.DataFrame.from_dict({
                'Delta_QV_Ah': QV_dQdV_cycleN['Capacity_Ah'] - QV_dQdV_cycle0['Capacity_Ah'],
                'Delta_dQdV_Ah/V': QV_dQdV_cycleN['dQdV_Ah/V'] - QV_dQdV_cycle0['dQdV_Ah/V'],
                })
    return extract_values

def get_delta_QV_feature(cell_data: CellData_MATR_LFPGr, voltage: float, cycle_0: int, transform: str = None) -> list:
    """
    Get the delta-QV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    if cycle_0 is None:
        idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    else:
        idx_cycle_1 = cycle_0
    extract_values = np.array(cell_data.measurement_extracts[f'discharge_QV_{int(voltage*1000)}mV'])
    extract_values = extract_values - extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_delta_dQdV_feature(cell_data: CellData_MATR_LFPGr, voltage: float, cycle_0: int, transform: str = None) -> list:
    """
    Get the delta-dQdV feature at a specific voltage from week N to week 0 for a given rate and voltage.
    """
    if cycle_0 is None:
        idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    else:
        idx_cycle_1 = cycle_0
    extract_values = np.array(cell_data.measurement_extracts[f'discharge_dQdV_{int(voltage*1000)}mV'])
    extract_values = extract_values - extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_relative_QV_feature(cell_data: CellData_MATR_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the relative-QV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    extract_values = np.array(cell_data.measurement_extracts[f'discharge_QV_{int(voltage*1000)}mV'])
    extract_values = extract_values / extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values.astype(float)

def get_relative_dQdV_feature(cell_data: CellData_MATR_LFPGr, voltage: float, transform: str = None) -> list:
    """
    Get the relative-dQdV feature at a specific voltage from week N to week 0 for a given rate and voltage.
    """
    idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    extract_values = np.array(cell_data.measurement_extracts[f'discharge_dQdV_{int(voltage*1000)}mV'])
    extract_values = extract_values / extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values.astype(float)

def get_delta_dQdV_integral_feature(cell_data: CellData_MATR_LFPGr, voltage_window: tuple, cycle_0: int, transform: str = None) -> list:
    """
    Get the delta-dQdV integral feature at a specific voltage window from week N to week 0 for a given rate and voltage.
    """
    if cycle_0 is None:
        idx_cycle_1 = np.argwhere(np.array(cell_data.charge_throughput_Ah) > 0)[0][0]
    else:
        idx_cycle_1 = cycle_0
    extract_values = np.array(cell_data.measurement_extracts[f"discharge_dQdV_integral_{int(voltage_window[0]*1000)}mV_{int(voltage_window[1]*1000)}mV"])
    extract_values = extract_values - extract_values[idx_cycle_1]
    extract_values[:idx_cycle_1] = np.nan
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs'")
    return extract_values.astype(float)

def get_delta_QV_dQdV_dVdQ_variable_statistic(cell_data: CellData_MATR_LFPGr, var: str, statistic: str, cycle_0: int) -> list:
    """
    Get a statistic of the delta-QV curve from week N to week 0.
    """
    if var not in ['Delta_QV_Ah', 'Delta_dQdV_Ah/V']:
        raise ValueError(f"Variable '{var}' is not supported. Supported variables are 'Delta_QV_Ah', 'Delta_dQdV_Ah/V'.")
    
    extract_tables = cell_data.measurement_extracts['discharge_delta_QV_dQdV']
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