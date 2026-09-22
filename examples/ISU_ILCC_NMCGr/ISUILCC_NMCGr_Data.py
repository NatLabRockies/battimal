import os
import pandas as pd
import numpy as np
import json
from battimal.data.battery_data import EchemMeasurement, CellData, BatteryDataset
from scipy.interpolate import make_splrep
import warnings

class RPT_ISUILCC_NMCGr(EchemMeasurement):
    """
    Class for keeping track of data from an electrochemical measurement.
    'IVSt' refers the the key variables in electrochemical data: current (I), voltage (V), step number (S), and 
    time (t). Optionally, other outputs from a battery test can be included, like power, charge throughput, ...
    This class also requires the type of measurement being stored, like 'formation' or 'RPT'. It is intended that 
    for any specific measurement type, there will be a subclass that inherits this to define specific attributes 
    and methods, such as unique plotting, or parsers to read the data in from a raw data file. Parsers could also
    be a defined function that outputs several EchemMeasurement objects, one for each measurement in the raw file.

    For ISU-ILCC-NMCGr, add capacity, energy, SOC variables, zero out time.
    Create extractors for different attributes. All extractors simply take in 'self' and return something else.
    """
    def __init__(self, measurement_type: str, this_rpt: dict):
        extractors = {
            'charge_capacity': get_charge_capacity,
            'charge_energy': get_charge_energy,
            'charge_cc_time': get_charge_cc_time,
            'charge_cv_time': get_charge_cv_time,
            'charge_cv_time_frac': get_charge_cv_time_frac,
            'discharge_capacity': get_discharge_capacity,
            'discharge_energy': get_discharge_energy,
            'coulombic_efficiency': get_coulombic_efficiency,
            'coulombic_inefficiency': lambda rpt: 1 - get_coulombic_efficiency(rpt),
            'log_coulombic_inefficiency': lambda rpt: np.log(1 - get_coulombic_efficiency(rpt)),
            'energy_efficiency': get_energy_efficiency,
            'energy_inefficiency': lambda rpt: 1 - get_energy_efficiency(rpt),
            'log_energy_inefficiency': lambda rpt: np.log(1 - get_energy_efficiency(rpt)),
            'charge_QV_dQdV_dVdQ': get_charge_QV_dQdV_dVdQ,         # 1000-point spline and derivative curves
            'discharge_QV_dQdV_dVdQ': get_discharge_QV_dQdV_dVdQ,   # 1000-point spline and derivative curves
            'DeltaV_SOC': get_DeltaV_SOC,                           # Splines of charge and discharge V(SOC) difference
        }
        # Assert type is either 'RPT C_5' or 'RPT C_2'
        assert measurement_type in ['RPT C_5', 'RPT C_2'], "Type must be either 'RPT C_5' or 'RPT C_2'"

        # Extractors for dQdV features at various voltages, dependent on rate
        if 'C_5' in measurement_type:
            discharge_dQdV_feature_voltages = [3.450, 3.580, 3.805, 3.860, 4.005]
            V_interpolation = np.linspace(3.0, 4.18, 1000)
        else: # C_2
            discharge_dQdV_feature_voltages = [3.425, 3.550, 3.815, 3.975]
            V_interpolation = np.linspace(3.0, 4.15, 1000)
        for voltage in discharge_dQdV_feature_voltages:
            index = np.argmin(np.abs(V_interpolation - voltage))
            extractors[f'discharge_dQdV_{int(voltage*1000)}mV'] = lambda rpt, index=index: rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index]['dQdV_Ah/V']
        # Extractors for QV features at various voltages, dependent on rate
        if 'C_5' in measurement_type:
            discharge_QV_feature_voltages = [3.420, 3.700, 4.000]
            V_interpolation = np.linspace(3.0, 4.18, 1000)
        else: # C_2
            discharge_QV_feature_voltages = [3.420, 3.650, 3.970]
            V_interpolation = np.linspace(3.0, 4.15, 1000)
        for voltage in discharge_QV_feature_voltages:
            index = np.argmin(np.abs(V_interpolation - voltage))
            extractors[f'discharge_QV_{int(voltage*1000)}mV'] = lambda rpt, index=index: rpt.extracts['discharge_QV_dQdV_dVdQ'].iloc[index]['Capacity_Ah']
        # Extractors for DeltaV(SOC) features at various SOCs
        soc_interpolate = np.linspace(0, 1, 1000)
        DeltaV_SOC_feature_SOCs = [0.03, 0.10, 0.30, 0.50, 0.70, 0.90]
        for soc in DeltaV_SOC_feature_SOCs:
            index = np.argmin(np.abs(soc_interpolate - soc))
            extractors[f'DeltaV_SOC_{int(soc*100)}SOC'] = lambda rpt, index=index: rpt.extracts['DeltaV_SOC'].iloc[index]['DeltaV']
        extractors['DeltaV_SOC_max'] = lambda rpt: np.max(rpt.extracts['DeltaV_SOC']['DeltaV'])
        extractors['DeltaV_SOC_SOC_at_max'] = lambda rpt: soc_interpolate[np.argmax(rpt.extracts['DeltaV_SOC']['DeltaV'])]

        # Init the EchemMeasurement object and add ISU-ILCC specific attributes
        super().__init__(measurement_type, this_rpt['I'], this_rpt['V'], this_rpt['step'], this_rpt['t'], extractors=extractors)
        self.Measurement_Time_s = get_rpt_time(this_rpt)
        self.Capacity_Ah = this_rpt['Q']
        self.Energy_Wh = this_rpt['E']
        self.SOC = get_soc(this_rpt)

    def to_df(self) -> pd.DataFrame:
        """
        Convert the RPT_ISUILCC_NMCGr data to a pandas DataFrame.
        :return: A pandas DataFrame with the measurement data.
        """
        return pd.DataFrame({
            'Current_A': self.Current_A,
            'Voltage_V': self.Voltage_V,
            'Step': self.Step,
            'Time_s': self.Time_s,
            'Measurement_Time_s': self.Measurement_Time_s,
            'Capacity_Ah': self.Capacity_Ah,
            'Energy_Wh': self.Energy_Wh,
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
            measurement_data = hdf5[measurement_key]
            this_rpt = {
                'I': measurement_data['Current_A'].to_list(),
                'V': measurement_data['Voltage_V'].to_list(),
                'step': measurement_data['Step'].to_list(),
                't': list(measurement_data['Time_s'].to_numpy()),
                'Q': measurement_data['Capacity_Ah'].to_list(),
                'E': measurement_data['Energy_Wh'].to_list(),
            }
            return RPT_ISUILCC_NMCGr(measurement_type, this_rpt)
        else:
            return RPT_ISUILCC_NMCGr(measurement_type, {
                'I': [],
                'V': [],
                'step': [],
                't': [],
                'Q': [],
                'E': []
            })
    

class CellData_ISUILCC_NMCGr(CellData):
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
        for rate in ['C_5', 'C_2']:
            measurement_extractors[f'relative_discharge_capacity_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_relative_discharge_capacity(cell_data, measurement_type)
            measurement_extractors[f'log_relative_discharge_capacity_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_relative_discharge_capacity(cell_data, measurement_type, transform)
            measurement_extractors[f'discharge_capacity_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_discharge_capacity_fade(cell_data, measurement_type)
            measurement_extractors[f'log_discharge_capacity_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_discharge_capacity_fade(cell_data, measurement_type, transform)
            measurement_extractors[f'relative_discharge_capacity_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_relative_discharge_capacity_fade(cell_data, measurement_type)
            measurement_extractors[f'log_relative_discharge_capacity_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_relative_discharge_capacity_fade(cell_data, measurement_type, transform)
            measurement_extractors[f'relative_discharge_energy_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_relative_discharge_energy(cell_data, measurement_type)
            measurement_extractors[f'log_relative_discharge_energy_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_relative_discharge_energy(cell_data, measurement_type, transform)
            measurement_extractors[f'discharge_energy_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_discharge_energy_fade(cell_data, measurement_type)
            measurement_extractors[f'log_discharge_energy_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_discharge_energy_fade(cell_data, measurement_type, transform)
            measurement_extractors[f'relative_discharge_energy_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_relative_discharge_energy_fade(cell_data, measurement_type)
            measurement_extractors[f'log_relative_discharge_energy_fade_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", transform='log': get_relative_discharge_energy_fade(cell_data, measurement_type, transform)

            measurement_extractors[f'charge_delta_QV_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", direction='charge': get_delta_QV_dQdV_table(cell_data, measurement_type, direction)
            measurement_extractors[f'discharge_delta_QV_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", direction='discharge': get_delta_QV_dQdV_table(cell_data, measurement_type, direction)
            measurement_extractors[f'delta_DeltaV_SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}": get_delta_DeltaV_SOC_table(cell_data, measurement_type)

            # Deltas, log(Deltas) of all QV, dQdV, and DeltaV(SOC) features
            if rate == 'C_5':
                discharge_dQdV_feature_voltages = [3.450, 3.580, 3.805, 3.860, 4.005]
                discharge_QV_feature_voltages = [3.420, 3.700, 4.000]
            else:
                discharge_dQdV_feature_voltages = [3.425, 3.550, 3.815, 3.975]
                discharge_QV_feature_voltages = [3.420, 3.650, 3.970]
            for voltage in discharge_dQdV_feature_voltages:
                measurement_extractors[f'discharge_delta_dQdV_{int(voltage*1000)}mV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", voltage=voltage: get_discharge_delta_dQdV_feature(cell_data, measurement_type, voltage)
                measurement_extractors[f'discharge_log_abs_delta_dQdV_{int(voltage*1000)}mV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", voltage=voltage, transform='log_abs': get_discharge_delta_dQdV_feature(cell_data, measurement_type, voltage, transform)
            for voltage in discharge_QV_feature_voltages:
                measurement_extractors[f'discharge_delta_QV_{int(voltage*1000)}mV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", voltage=voltage: get_discharge_delta_QV_feature(cell_data, measurement_type, voltage)
                measurement_extractors[f'discharge_log_abs_delta_QV_{int(voltage*1000)}mV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", voltage=voltage, transform='log_abs': get_discharge_delta_QV_feature(cell_data, measurement_type, voltage, transform)
            DeltaV_SOC_feature_SOCs = [0.03, 0.10, 0.30, 0.50, 0.70, 0.90]
            for soc in DeltaV_SOC_feature_SOCs:     
                measurement_extractors[f'delta_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", soc=soc: get_delta_DeltaV_SOC_feature(cell_data, measurement_type, soc)
                measurement_extractors[f'log_abs_delta_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", soc=soc, transform='log_abs': get_delta_DeltaV_SOC_feature(cell_data, measurement_type, soc, transform)
                measurement_extractors[f'relative_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", soc=soc: get_relative_DeltaV_SOC(cell_data, measurement_type, soc)
                measurement_extractors[f'log_relative_DeltaV_SOC_{int(soc*100)}SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", soc=soc, transform='log': get_relative_DeltaV_SOC(cell_data, measurement_type, soc, transform)

            for direction in ['discharge', 'charge']:
                measurement_extractors[f'{direction}_var_delta_QV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='var': get_delta_QV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_mean_delta_QV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='mean': get_delta_QV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_log_var_delta_QV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='log_var': get_delta_QV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_log_abs_mean_delta_QV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='log_abs_mean': get_delta_QV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_var_delta_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='var': get_delta_dQdV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_mean_delta_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='mean': get_delta_dQdV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_log_var_delta_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='log_var': get_delta_dQdV_statistic(cell_data, measurement_type, rate, direction, statistic)
                measurement_extractors[f'{direction}_log_abs_mean_delta_dQdV_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, direction=direction, statistic='log_abs_mean': get_delta_dQdV_statistic(cell_data, measurement_type, rate, direction, statistic)
            measurement_extractors[f'var_delta_DeltaV_SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, statistic='var': get_delta_DeltaV_SOC_statistic(cell_data, measurement_type, rate, statistic)
            measurement_extractors[f'mean_delta_DeltaV_SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, statistic='mean': get_delta_DeltaV_SOC_statistic(cell_data, measurement_type, rate, statistic)
            measurement_extractors[f'log_var_delta_DeltaV_SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, statistic='log_var': get_delta_DeltaV_SOC_statistic(cell_data, measurement_type, rate, statistic)
            measurement_extractors[f'log_mean_delta_DeltaV_SOC_{rate}'] = lambda cell_data, measurement_type=f"RPT {rate}", rate=rate, statistic='log_mean': get_delta_DeltaV_SOC_statistic(cell_data, measurement_type, rate, statistic)
        # # remaining useful life target and week feature
        measurement_extractors['remaining_weeks_to_80percent_C_5_capacity'] = get_rul
        measurement_extractors['log_remaining_weeks_to_80percent_C_5_capacity'] = lambda cell_data, transform='log': get_rul(cell_data, transform)
        measurement_extractors['num_week'] = lambda cell_data: np.array(cell_data.repeat_num) - 1 # first repeat is number 1 which corresponds to week 0

        # ISU-ILCC cell-level extractors
        cell_extractors = {
            'weeks_to_80percent_C_5_capacity': lambda cell_data: get_weeks_to_failure(cell_data, 0.8),
            'log_weeks_to_80percent_C_5_capacity': lambda cell_data: np.log(get_weeks_to_failure(cell_data, 0.8)),
        }

        super().__init__(cell_id, 
                         cell_metadata=cell_metadata, 
                         characterization_data=characterization_data, 
                         measurement_extractors=measurement_extractors, 
                         cell_extractors=cell_extractors)
    
    def append(self, data: EchemMeasurement, t_0: float = 0, repeat_modifier: int = 0, cycle_modifier: int = 0):
        try:
            start_time_s = (data.Time_s[0] - t_0) / np.timedelta64(1, "s")
        except:
            start_time_s = 0 # breaks when loading from h5, but we overwrite anyways so we don't care
        cycles = 1 + cycle_modifier
        Q_dis = get_discharge_capacity(data)
        Q_chg = get_charge_capacity(data)
        charge_throughput_Ah = Q_dis + Q_chg
        return super().append(data, 
                              start_time_s=start_time_s, 
                              cycles=cycles, 
                              repeat_modifier=repeat_modifier, 
                              charge_throughput_Ah=charge_throughput_Ah
                              )

class Dataset_ISUILCC_NMCGr(BatteryDataset):
    """
    ISU ILCC NMC-Gr Preprocessor
    """

    def __init__(self,
                 battery_data: list[CellData] = None,
                 metadata: dict = None,
                 data_fpath: str = os.path.join('data_raw', 'RPT_json', 'RPT_json'), 
                 metadata_fpath: str = os.path.join('data_raw', 'C_rates_and_DoD.csv'),
                 is_short: bool = False, # if True, only load a small subset of the data for testing
        ):
        """
        data_fpath: path to the raw data files
        """

        if battery_data is not None:
            self.battery_data = battery_data
            self.metadata = metadata

        else:
            # Ignore all RuntimeWarnings (repeated divide by zero warnings which we can ignore)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Cell metadata
            cell_metadata = pd.read_csv(metadata_fpath).drop(columns=['Unnamed: 0'])
            cell_metadata['Stress_charge'] = cell_metadata['mean_C_chg']**0.5 * cell_metadata['DoD']**0.5
            cell_metadata['Stress_discharge'] = cell_metadata['mean_C_dis'].abs()**0.5 * cell_metadata['DoD']**0.5
            cell_metadata['Stress_avg'] = cell_metadata[['Stress_charge', 'Stress_discharge']].mean(axis=1)
            
            shortlist_cells = ["G1C1",
                                "G2C4",
                                "G49C3",
                                "G50C3"]
            
            # Load each RPT of each cell
            batch2 = [
                "G57C1",
                "G57C2",
                "G57C3",
                "G57C4",
                "G58C1",
                "G26C3",
                "G49C1",
                "G49C2",
                "G49C3",
                "G49C4",
                "G50C1",
                "G50C3",
                "G50C4",
            ]
            # for every '.json' in 'Release 1.0' and 'Release 2.0' folders, load rpts, check for noisy/missing data, get extracts, save to hdf5
            dataset = []
            for dirpath, dirnames, files in os.walk(data_fpath):
                for file in files:
                    if '.json' in file:
                        cell = file.split(".")[0]
                        group = int(cell.split("C")[0][1:])
                        if is_short and cell not in shortlist_cells:
                            continue

                        print("Processing cell " + cell)
                        if cell not in batch2:
                            subfolder = "Release 1.0"
                        else:
                            subfolder = "Release 2.0"
                        metadata_this_cell = cell_metadata[cell_metadata['cell'] == cell].drop(columns=['cell']).to_dict(orient='records')[0]
                        metadata_this_cell['GroupNumber'] = group
                        cell_data = CellData_ISUILCC_NMCGr(cell, cell_metadata=metadata_this_cell)
                        rpts = load_cell_rpts(data_fpath, cell, subfolder)
                        n_rpts = len(rpts['QV_charge_C_5']['V'])
                        repeat_modifier_C_5, repeat_modifier_C_2, cycle_modifier = 0, 0, 0
                        for i in range(n_rpts):
                            try:
                                # C/5 charge and discharge
                                this_rpt_C_5 = assemble_rpt_data(rpts, 'C_5', i)
                                if i == 0:
                                    t_0 = this_rpt_C_5['t'][0]
                                is_bad_data, failure_case = check_for_missing_or_noisy_data(this_rpt_C_5)
                                if is_bad_data:
                                    print(f"Cell {cell} RPT C_5 {i+1} bad data, failure: {failure_case}.")
                                    repeat_modifier_C_5 += 1
                                    cycle_modifier += 1
                                else:
                                    cell_data.append(RPT_ISUILCC_NMCGr("RPT C_5", this_rpt_C_5), t_0=t_0, repeat_modifier=repeat_modifier_C_5, cycle_modifier=cycle_modifier)
                                    cycle_modifier = 0
                                # C/2 charge and discharge
                                this_rpt_C_2 = assemble_rpt_data(rpts, 'C_2', i)
                                is_bad_data, failure_case = check_for_missing_or_noisy_data(this_rpt_C_2)
                                if is_bad_data:
                                    print(f"Cell {cell} RPT C_2 {i+1} bad data, failure: {failure_case}")
                                    repeat_modifier_C_2 += 1
                                    cycle_modifier += 1
                                else:
                                    cell_data.append(RPT_ISUILCC_NMCGr("RPT C_2", this_rpt_C_2), t_0=t_0, repeat_modifier=repeat_modifier_C_2, cycle_modifier=cycle_modifier)
                                    cycle_modifier = 0
                            except:
                                print(f"Cell {cell} RPT {i+1} missing")
                                repeat_modifier_C_5 += 1
                                repeat_modifier_C_2 += 1
                                cycle_modifier += 2
                        try:
                            cell_data.run_extractors()
                            dataset.append(cell_data)
                        except:
                            print(f"!~!~!~!~!    Cell {cell} failed to run extractors            !~!~!~!~!")
                            print(f"!~!~!~!~!    Cell {cell} will not be saved to the dataset    !~!~!~!~!")
                        
                        del cell_data
                    
                # Init BatteryDataset 
                super().__init__(dataset, metadata={
                    'name': 'ISU_ILCC_NMCGr',
                    'description': 'ISU ILCC NMC-Gr Dataset',
                    'target_var': 'log_weeks_to_80percent_C_5_capacity',
                    'decision_variables_univariate': 'Stress_avg',
                    'decision_variables_multivariate': ['mean_C_chg', 'mean_C_dis', 'DoD'],
                    'trajectory_variable': 'repeat_num',
                    'splitting_test_description':  "By 'GroupNumber'",
                    'splitting_train_description': "By 'GroupNumber'",
                    'splitting_val_description':   "By 'GroupNumber'",
                    })
        
    @staticmethod
    def from_h5(hdf5: pd.HDFStore, ignore_measurement_data: bool = False, ignore_high_dim_measurement_extracts: bool = False):
        return BatteryDataset.from_h5(Dataset_ISUILCC_NMCGr,
                                      hdf5,
                                      celldata_classes={'CellData_ISUILCC_NMCGr': CellData_ISUILCC_NMCGr},
                                      measurement_classes={'RPT_ISUILCC_NMCGr': RPT_ISUILCC_NMCGr},
                                      ignore_measurement_data=ignore_measurement_data,
                                      ignore_high_dim_measurement_extracts=ignore_high_dim_measurement_extracts
                                      )

    def get_failure_matrices(self, target_var_name='log_weeks_to_80percent_C_5_capacity', measurement_type='RPT C_5', alignment_var='repeat_num',
                             cumulative=False, nan_filter_method='drop_column', n_skip_measurements=1, force_consistent_vars=True):
        cell_ids, xy_pairs = super().get_failure_matrices(target_var_name, measurement_type, alignment_var,
                                                          cumulative, nan_filter_method, n_skip_measurements, force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_weeks_to_80percent_C_5_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs

    def get_rul_matrices(self, target_var_name='log_remaining_weeks_to_80percent_C_5_capacity', measurement_type='RPT C_5', nan_filter_method='drop_column', n_skip_measurements=1):
        cell_ids, (x, y) = super().get_rul_matrices(target_var_name, measurement_type, nan_filter_method, n_skip_measurements)
        for col in x.columns:
            if 'remaining_weeks_to_80percent_C_5_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_lookahead_matrices(self, target_var_name='relative_discharge_capacity_C_5', measurement_type='RPT C_5', alignment_var='repeat_num',
                               cumulative=False, nan_filter_method='drop_column', n_skip_measurements=1, force_consistent_vars=True):
        cell_ids, xy_pairs = super().get_lookahead_matrices(target_var_name, measurement_type, alignment_var, cumulative, nan_filter_method, n_skip_measurements, force_consistent_vars)
        # drop remaining useful life target to prevent data leakage
        for i, xy in enumerate(xy_pairs):
            x = xy[0]
            for col in x.columns:
                if 'remaining_weeks_to_80percent_C_5_capacity' in col:
                    x = x.drop(columns=[col])
                    xy_pairs[i] = (x, xy[1])
        return cell_ids, xy_pairs
    
    def get_autoregressive_matrices(self, target_var_name='relative_discharge_capacity_C_5', measurement_type='RPT C_5', n_lags=1, is_diff_x=True, is_diff_y=True, nan_filter_method='drop_column', n_skip_measurements=1):
        cell_ids, (x, y) = super().get_autoregressive_matrices(target_var_name=target_var_name, measurement_type=measurement_type,
                                                               n_lags=n_lags, is_diff_x=is_diff_x, is_diff_y=is_diff_y, 
                                                               nan_filter_method=nan_filter_method, n_skip_measurements=n_skip_measurements)
        # drop remaining useful life target and any lags of it to prevent data leakage
        for col in x.columns:
            if 'remaining_weeks_to_80percent_C_5_capacity' in col:
                x = x.drop(columns=[col])
        return cell_ids, (x, y)
    
    def get_trajectory_matrices(self, target_var_name='relative_discharge_capacity_C_5', chronological_var_name='repeat_num', measurement_type='RPT C_5', n_skip_measurements=0):
        return super().get_trajectory_matrices(target_var_name, chronological_var_name, measurement_type, n_skip_measurements)


"""
ISU-ILCC NMC-Gr data processing functions
"""
def load_cell_rpts(data_fpath, cell, subfolder):
    # Read RPTs from a single cell into a dictionary, changing formatting of the time column
    file_path = os.path.join(data_fpath, subfolder, f"{cell}.json")
    with open(file_path, "r") as file:
        data_dict = json.loads(json.load(file))

    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict["start_stop_time"]["start"]):
        if start_time != "[]":
            data_dict["start_stop_time"]["start"][iii] = np.datetime64(start_time)
            data_dict["start_stop_time"]["stop"][iii] = np.datetime64(
                data_dict["start_stop_time"]["stop"][iii]
            )
        else:
            data_dict["start_stop_time"]["start"][iii] = []
            data_dict["start_stop_time"]["stop"][iii] = []

    for iii in range(len(data_dict["start_stop_time"]["start"])):
        data_dict["QV_charge_C_2"]["t"][iii] = list(
            map(np.datetime64, data_dict["QV_charge_C_2"]["t"][iii])
        )
        data_dict["QV_discharge_C_2"]["t"][iii] = list(
            map(np.datetime64, data_dict["QV_discharge_C_2"]["t"][iii])
        )
        data_dict["QV_charge_C_5"]["t"][iii] = list(
            map(np.datetime64, data_dict["QV_charge_C_5"]["t"][iii])
        )
        data_dict["QV_discharge_C_5"]["t"][iii] = list(
            map(np.datetime64, data_dict["QV_discharge_C_5"]["t"][iii])
        )

    return data_dict

def assemble_rpt_data(rpts, rate, i):
    # concatenate charge and discharge data from this rpt for each variable
    # assign step numbers to CC and CV portions of charge
    # assert that rate is 'C_5' or 'C_2'
    assert rate in ['C_5', 'C_2'], "Rate must be either 'C_5' or 'C_2'"
    vars = ['Q', 'V', 'I', 'E', 't']
    rpt_data = {}
    for var in vars:
        rpt_data[var] = rpts["QV_charge_" + rate][var][i]
    # All discharges are single step (only CC phase). Charges have CC-CV phases
    rpt_data['step'] = get_charge_steps(rpt_data)
    for var in vars:
        rpt_data[var].extend(rpts["QV_discharge_" + rate][var][i])
    rpt_data['step'].extend([3] * len(rpts["QV_discharge_" + rate][var][i]))
    # Calculate charge throughput
    Q_dis = rpt_data['Q'][-1]
    idx_dis_start = np.argwhere(np.array(rpt_data['step']) == 3)[0][0]
    Q_chg = rpt_data['Q'][idx_dis_start-1]
    rpt_data['charge_throughput_Ah'] = Q_dis + Q_chg
    return rpt_data

def get_charge_steps(rpt):
    step = []
    current = np.array(rpt['I']) 
    # sometimes there's a current ramp up from 0 in the first few seconds. Only search high V
    idx_4V = np.argwhere(np.array(rpt['V']) > 4.1)[0][0]
    # atol = 1e-3: step ends too early sometimes
    # atol = 5e-3: step gets first data point of CV phase
    # atol = 2e-3: a bit better, at least doesn't end too early
    # moved voltage to 4.1V, removed rtol: doesn't end early but not perfect
    # atol = 1.5e-3: very close to perfect
    # atol = 1.25e-3: too small
    idx = np.argwhere(~np.isclose(current[idx_4V:], current[idx_4V], atol=1.5e-3, rtol=0))[0][0] 
    idx += idx_4V
    if rpt['V'][idx] < 4.18:
        # print("WARNING: a charge CC step ends before 4.18V")
        scalar = 1.5
        atol = 1.5e-3 * scalar
        max_iters = 6
        iter = 0
        while rpt['V'][idx] < 4.18:
            # print("increasing tolerance....")
            idx = np.argwhere(~np.isclose(current[idx_4V:], current[idx_4V], atol=atol, rtol=0))[0][0] 
            idx += idx_4V
            iter += 1
            atol = atol * scalar
            if iter > max_iters:
                break

    step = [1] * idx
    step.extend([2] * (len(rpt['I']) - idx))
    return step

def get_rpt_time(rpt):
    if len(rpt['t']) > 0:
        # zero out time
        t = (rpt['t'] - rpt['t'][0]) / np.timedelta64(1, "s")
        # sometimes time goes backwards during discharge, fix it
        t = np.array(t)
        dt = np.diff(t, prepend=0)
        dt[dt < 0] = np.median(dt[dt > 0])
        dt[dt > 1e8] = np.median(dt[dt > 0]) # one gross outlier in an rpt from G57C2
        t = np.cumsum(dt)
        return t.tolist()
    else:
        return []

def get_soc(rpt):
    if len(rpt['Q']) > 0:
        q = np.array(rpt['Q'])
        soc = np.zeros(len(q))
        idx_dis_start = np.argwhere(np.array(rpt['step']) == 3)[0][0]
        q_chg = q[idx_dis_start-1]
        q_dis = q[-1]
        soc[:idx_dis_start] = q[:idx_dis_start] / q_chg
        soc[idx_dis_start:] = q[idx_dis_start:] / q_dis
        return soc.tolist()
    else:
        return []

def check_for_missing_or_noisy_data(rpt, tol_res=4e-5):
    is_bad_data = False
    failure_case = None
    # Check for noisy data
    for step in [1, 3]: # charge and discharge CC steps
        # polyfit the Q-V curve for each step
        this_step = np.argwhere(np.array(rpt['step']) == step).squeeze()
        t = get_rpt_time(rpt)
        Q = rpt['Q'][this_step[0]:this_step[-1]]
        V = rpt['V'][this_step[0]:this_step[-1]]
        # ignore voltages less than 3.3 to avoid asympote
        mask = np.argwhere(np.array(V) > 3.3).squeeze().tolist()
        Q = Q[mask[0]:mask[-1]]
        V = V[mask[0]:mask[-1]]

        # fit a polynomial to the data
        coeffs = np.polyfit(Q, V, 10)
        poly = np.poly1d(coeffs)
        V_fit = poly(Q)
        residuals = V_fit - V
        # Avoid noisy fit at low SOC, due to asymptotic behavior of the voltage that isn't nicely modeled by a polynomial
        residuals = residuals[np.array(Q) > 0.01]
        if np.var(residuals) > tol_res:
             is_bad_data = True
             failure_case = 'noisy'
            #  line = plt.plot(Q, V, '-')
            #  plt.plot(Q, V_fit, '--', color=line[0].get_color(), label=np.var(residuals))
            #  plt.legend()

    # Check for missing data - some discharges are missing the end
    this_step = np.argwhere(np.array(rpt['step']) == 3).squeeze()
    V = rpt['V'][this_step[0]:this_step[-1]]
    if min(V) > 3.05:
        is_bad_data = True
        failure_case = 'missing'
    
    return is_bad_data, failure_case

"""
Measurement extractors
"""
def get_charge_capacity(rpt: RPT_ISUILCC_NMCGr) -> float:
    if len(rpt.Capacity_Ah) > 0:
        idx_dis_start = np.argwhere(np.array(rpt.Step) == 3)[0][0]
        return rpt.Capacity_Ah[idx_dis_start-1]
    else:
        return 0.0

def get_charge_energy(rpt: RPT_ISUILCC_NMCGr) -> float:
    idx_dis_start = np.argwhere(np.array(rpt.Step) == 3)[0][0]
    return rpt.Energy_Wh[idx_dis_start-1]

def get_charge_cc_time(rpt: RPT_ISUILCC_NMCGr) -> float:
    idx_cc = np.argwhere(np.array(rpt.Step) == 1).squeeze()
    return rpt.Measurement_Time_s[idx_cc[-1]] - rpt.Measurement_Time_s[idx_cc[0]]
    
def get_charge_cv_time(rpt: RPT_ISUILCC_NMCGr) -> float:
    idx_cv = np.argwhere(np.array(rpt.Step) == 2).squeeze()
    return rpt.Measurement_Time_s[idx_cv[-1]] - rpt.Measurement_Time_s[idx_cv[0]]

def get_charge_cv_time_frac(rpt: RPT_ISUILCC_NMCGr) -> float:
    return get_charge_cv_time(rpt) / (get_charge_cc_time(rpt) + get_charge_cv_time(rpt))

def get_discharge_capacity(rpt: RPT_ISUILCC_NMCGr) -> float:
    if len(rpt.Capacity_Ah) > 0:
        return rpt.Capacity_Ah[-1]
    else:
        return 0.0

def get_discharge_energy(rpt: RPT_ISUILCC_NMCGr) -> float:
    return rpt.Energy_Wh[-1]

def get_coulombic_efficiency(rpt: RPT_ISUILCC_NMCGr) -> float:
    return get_discharge_capacity(rpt) / get_charge_capacity(rpt)

def get_energy_efficiency(rpt: RPT_ISUILCC_NMCGr) -> float:
    return get_discharge_energy(rpt) / get_charge_energy(rpt)

def get_charge_QV_dQdV_dVdQ(rpt: RPT_ISUILCC_NMCGr) -> pd.DataFrame:
    # Q(V) spline for CC charge (CV is const. voltage, can't spline)
    if 'C_5' in rpt.measurement_type:
        V_interpolation = np.linspace(3.1, 4.2, 1000)
    else: # C_2
        V_interpolation = np.linspace(3.3, 4.2, 1000)
    idx_cc = np.argwhere(np.array(rpt.Step) == 1).squeeze()
    Q_raw = np.array(rpt.Capacity_Ah)[idx_cc]
    V_raw = np.array(rpt.Voltage_V)[idx_cc]
    if V_raw[-1] >= 4.2: # could be one point from CV phase here
        mask = V_raw < 4.2
        V_raw = V_raw[mask]
        Q_raw = Q_raw[mask]
    if V_raw[0] < V_interpolation[0]:
        V_raw = np.concatenate(([V_interpolation[0]], V_raw))
        Q_raw = np.concatenate(([0], Q_raw))
    # set spline precision
    if len(Q_raw) >= 10000:
        precision = 8e-4*5
    else:
        precision = 2e-5*5
    idx_sort = np.argsort(V_raw)
    V_raw, Q_raw = V_raw[idx_sort], Q_raw[idx_sort]
    if len(np.unique(V_raw)) < len(V_raw):
        V_raw, idx_unique = np.unique(V_raw, return_index=True)
        Q_raw = Q_raw[idx_unique]
    spline = make_splrep(V_raw, Q_raw, s=precision)
    spline_derivative = spline.derivative()

    qv_new = spline(V_interpolation)
    dqdv_new = spline_derivative(V_interpolation)
    dvdq_new = 1 / dqdv_new
    # QV measurements don't have prior voltage point included and can start at unexpected values
    # due to high initial overpotential, causing weird spline behavior in extrapolation
    if np.any(qv_new < 0):
        mask_fix = qv_new < 0
        idx_fix = np.max(np.argwhere(mask_fix))
        qv_new[0:idx_fix] = 0
        dqdv_new[0:idx_fix] = 0
        dvdq_new[0:idx_fix] = np.nan
    
    QV_spline = {
        'Voltage_V': V_interpolation,
        'Capacity_Ah': qv_new,
        'dQdV_Ah/V': dqdv_new,
        'dVdQ_V/Ah': dvdq_new,
    }
    return pd.DataFrame(QV_spline)

def get_discharge_QV_dQdV_dVdQ(rpt: RPT_ISUILCC_NMCGr) -> pd.DataFrame:
    # Q(V) spline for CC discharge
    if 'C_5' in rpt.measurement_type:
        V_interpolation = np.linspace(3.0, 4.18, 1000)
    else: # C_2
        V_interpolation = np.linspace(3.0, 4.15, 1000)
    idx_dis = np.argwhere(np.array(rpt.Step) == 3).squeeze()
    Q_raw = np.array(rpt.Capacity_Ah)[idx_dis]
    V_raw = np.array(rpt.Voltage_V)[idx_dis]
    if V_raw[0] < V_interpolation[-1]:
        V_raw = np.concatenate(([V_interpolation[-1]], V_raw))
        Q_raw = np.concatenate(([0], Q_raw))
    if len(Q_raw) >= 10000:
        precision = 8e-4
    else:
        precision = 2e-5
    idx_sort = np.argsort(V_raw)
    V_raw, Q_raw = V_raw[idx_sort], Q_raw[idx_sort]
    if len(np.unique(V_raw)) < len(V_raw):
        V_raw, idx_unique = np.unique(V_raw, return_index=True)
        Q_raw = Q_raw[idx_unique]
    spline = make_splrep(V_raw, Q_raw, s=precision)
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
        'dQdV_Ah/V': dqdv_new,
        'dVdQ_V/Ah': dvdq_new,
    }
    return pd.DataFrame.from_dict(QV_spline)

def get_DeltaV_SOC(rpt: RPT_ISUILCC_NMCGr) -> pd.DataFrame:
    soc_interpolate = np.linspace(0, 1, 1000)
    idx_dis_start = np.argwhere(np.array(rpt.Step) == 3)[0][0]
    # charge
    soc_chg = rpt.SOC[:idx_dis_start]
    V_chg = rpt.Voltage_V[:idx_dis_start]
    if soc_chg[1] == 0:
        soc_chg = soc_chg[1:]  # avoid duplicate 0
        V_chg = V_chg[1:] 
    spline_chg = make_splrep(soc_chg, V_chg)
    # discharge
    soc_dis = rpt.SOC[idx_dis_start:]
    V_dis = rpt.Voltage_V[idx_dis_start:]
    if soc_dis[1] == 0:
        soc_dis = soc_dis[1:]  # avoid duplicate 0
        V_dis = V_dis[1:] 
    spline_dis = make_splrep(soc_dis, V_dis)
    DeltaV_SOC = {
        'SOC': soc_interpolate,
        'DeltaV': spline_chg(soc_interpolate) - spline_dis(soc_interpolate)[::-1],
    }
    return pd.DataFrame.from_dict(DeltaV_SOC)

"""
Cell-level extractors
"""
def get_weeks_to_failure(cell_data: CellData_ISUILCC_NMCGr, threshold: float, measurement_type: str = 'RPT C_5') -> int:
    """
    Get the number of weeks until the cell reaches a certain threshold of capacity.
    """
    relative_discharge_capacity = np.array(cell_data.measurement_extracts['relative_discharge_capacity_C_5'])
    rpt_week = np.array(cell_data.repeat_num)
    index = np.argwhere(relative_discharge_capacity != None) # RPT C_2 measurements or missing data will have None values
    relative_discharge_capacity = relative_discharge_capacity[index]
    rpt_week = rpt_week[index]
    # Find the index of the first value below the threshold
    index = np.argmax(relative_discharge_capacity < threshold)
    if index is None:
        ValueError("Couldn't find weeks to failure!")
    else:
        return int(rpt_week[index] - 1) # first repeat is number 1 which corresponds to week 0

"""
Measurement extractors at cell-level (requires differences or comparisons between measurements)
"""
def get_relative_discharge_capacity(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, transform: str = None) -> CellData_ISUILCC_NMCGr:
    """
    Get the relative discharge capacity for a given rate.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_capacity', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values / extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_rul(cell_data: CellData_ISUILCC_NMCGr, transform: str = None) -> CellData_ISUILCC_NMCGr:
    """
    Get remaining weeks (RPT repeats) until the cell hits the 80 percent relative capacity
    end of life threshold at C_5 rate
    """
    
    relative_discharge_capacity = np.array(cell_data.measurement_extracts['relative_discharge_capacity_C_5'])
    weeks = np.array(cell_data.repeat_num) - 1. # first repeat is number 1 which corresponds to week 0
    weeks_to_failure = get_weeks_to_failure(cell_data, 0.8)
    remaining_weeks = weeks_to_failure - weeks
    remaining_weeks[remaining_weeks < 0] = np.nan
    remaining_weeks[np.isnan(relative_discharge_capacity)] = np.nan
    if transform == 'log':
        remaining_weeks[remaining_weeks <= 0] = np.nan
        remaining_weeks = np.log(remaining_weeks)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return remaining_weeks


def get_discharge_capacity_fade(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, transform: str = None) -> CellData_ISUILCC_NMCGr:
    """
    Get the discharge capacity fade from week N to week 0 for a given rate.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_capacity', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values[np.argwhere(mask_this_measurement)[0][0]] - extract_values
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_relative_discharge_capacity_fade(cell_data: CellData_ISUILCC_NMCGr, measurement_type, transform: str=None) -> CellData_ISUILCC_NMCGr:
    """
    Get the relative discharge capacity fade from week N to week 0 for a given rate.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_capacity', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = (extract_values[np.argwhere(mask_this_measurement)[0][0]] - extract_values) / extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
       if transform is not None:
             raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_relative_discharge_energy(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, transform: str = None) -> CellData_ISUILCC_NMCGr:
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_energy', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values / extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_discharge_energy_fade(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, transform: str = None) -> CellData_ISUILCC_NMCGr:
    """
    Get the discharge energy fade from week N to week 0 for a given rate.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_energy', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values[np.argwhere(mask_this_measurement)[0][0]] - extract_values
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_relative_discharge_energy_fade(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, transform: str = None) -> CellData_ISUILCC_NMCGr:
    """
    Get the relative discharge energy fade from week N to week 0 for a given rate.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type('discharge_energy', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = (extract_values[np.argwhere(mask_this_measurement)[0][0]] - extract_values) / extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log'")
    return extract_values

def get_delta_QV_dQdV_table(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, direction: str) -> CellData_ISUILCC_NMCGr:
    """
    Get the delta-QV and delta-dQdV curves from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts[f'{direction}_QV_dQdV_dVdQ']
    mask_this_measurement = cell_data.get_measurement_type_mask(measurement_type)
    extract_values = [None] * len(extract_tables)
    for index, (QV_dQdV_weekN, is_this_measurement_type) in enumerate(zip(extract_tables, mask_this_measurement)):
        if is_this_measurement_type:
            if cell_data.repeat_num[index] == 1:
                QV_dQdV_week0 = QV_dQdV_weekN
            else:
                extract_values[index] = pd.DataFrame.from_dict({
                    'Voltage_V': QV_dQdV_week0['Voltage_V'],
                    'Delta_QV_Ah': QV_dQdV_weekN['Capacity_Ah'] - QV_dQdV_week0['Capacity_Ah'],
                    'Delta_dQdV_Ah/V': QV_dQdV_weekN['dQdV_Ah/V'] - QV_dQdV_week0['dQdV_Ah/V'],
                    })
    return extract_values

def get_discharge_delta_dQdV_feature(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, voltage: float, transform: str = None) -> list:
    """
    Get the delta-dQdV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type(f'discharge_dQdV_{int(voltage*1000)}mV', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values - extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs' for discharge delta-dQdV feature.")
    return extract_values.astype(float)

def get_discharge_delta_QV_feature(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, voltage: float, transform: str = None) -> list:
    """
    Get the delta-QV feature at a specific voltage and direction from week N to week 0 for a given rate and voltage.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type(f'discharge_QV_{int(voltage*1000)}mV', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values - extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log_abs' for discharge delta-QV feature.")
    return extract_values.astype(float)

def get_delta_DeltaV_SOC_table(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str) -> list:
    """
    Get the delta-DeltaV(SOC) curves from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts['DeltaV_SOC']
    mask_this_measurement = cell_data.get_measurement_type_mask(measurement_type)
    extract_values = [None] * len(extract_tables)
    for index, (DeltaV_SOC_weekN, is_this_measurement_type) in enumerate(zip(extract_tables, mask_this_measurement)):
        if is_this_measurement_type:
            if cell_data.repeat_num[index] == 1:
                DeltaV_SOC_week0 = DeltaV_SOC_weekN.copy()
            else:
                extract_values[index] = pd.DataFrame.from_dict({
                    'SOC': DeltaV_SOC_week0['SOC'],
                    'delta_DeltaV': DeltaV_SOC_weekN['DeltaV'] - DeltaV_SOC_week0['DeltaV'],
                })
    return extract_values

def get_delta_DeltaV_SOC_feature(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, soc: float, transform: str = None) -> list:
    """
    Get the delta-DeltaV_SOC feature at a specific SOC and direction from week N to week 0 for a given rate and SOC.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type(f'DeltaV_SOC_{int(soc*100)}SOC', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values - extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log_abs':
        extract_values = np.log(np.abs(extract_values))
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log' for delta DeltaV-SOC feature.")
    return extract_values

def get_relative_DeltaV_SOC(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, soc: float, transform: str = None) -> list:
    """
    Get the relative DeltaV_SOC feature at a specific SOC and direction from week N to week 0 for a given rate and SOC.
    """
    (extract_values, mask_this_measurement) = cell_data.get_extract_array_by_measurement_type(f'DeltaV_SOC_{int(soc*100)}SOC', measurement_type)
    extract_values = extract_values.astype(float)
    extract_values = extract_values / extract_values[np.argwhere(mask_this_measurement)[0][0]]
    if transform == 'log':
        extract_values = np.log(extract_values)
    else:
        if transform is not None:
            raise ValueError("Transform must be None or 'log' for relative DeltaV feature.")
    return extract_values

def get_delta_QV_statistic(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, rate: str, direction: str, statistic: str) -> list:
    """
    Get a statistic of the delta-QV curve from week N to week 0 for a given rate and direction.
    """
    extract_tables = cell_data.measurement_extracts[f'{direction}_delta_QV_dQdV_{rate}']
    mask_this_measurement = cell_data.get_measurement_type_mask(measurement_type)
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, (Delta_QV_dQdV_dVdQ, is_this_measurement_type) in enumerate(zip(extract_tables, mask_this_measurement)):
        if is_this_measurement_type:
            if cell_data.repeat_num[index] > 1:
                if statistic == 'var':
                    stat_value = np.var(Delta_QV_dQdV_dVdQ['Delta_QV_Ah'])
                elif statistic == 'mean':
                    stat_value = np.mean(Delta_QV_dQdV_dVdQ['Delta_QV_Ah'])
                elif statistic == 'log_var':
                    stat_value = np.log(np.var(Delta_QV_dQdV_dVdQ['Delta_QV_Ah']))
                elif statistic == 'log_abs_mean':
                    stat_value = np.log(np.abs(np.mean(Delta_QV_dQdV_dVdQ['Delta_QV_Ah'])))
                else:
                    raise ValueError(f"Statistic '{statistic}' is not supported.")
                extract_values[index] = stat_value
    return extract_values

def get_delta_dQdV_statistic(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, rate: str, direction: str, statistic: str) -> list:
    """
    Get a statistic of the delta-dQdV curve from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts[f'{direction}_delta_QV_dQdV_{rate}']
    mask_this_measurement = cell_data.get_measurement_type_mask(measurement_type)
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, (Delta_QV_dQdV_dVdQ, is_this_measurement_type) in enumerate(zip(extract_tables, mask_this_measurement)):
        if is_this_measurement_type:
            if cell_data.repeat_num[index] > 1:
                if statistic == 'var':
                    stat_value = np.var(Delta_QV_dQdV_dVdQ['Delta_dQdV_Ah/V'])
                elif statistic == 'mean':
                    stat_value = np.mean(Delta_QV_dQdV_dVdQ['Delta_dQdV_Ah/V'])
                elif statistic == 'log_var':
                    stat_value = np.log(np.var(Delta_QV_dQdV_dVdQ['Delta_dQdV_Ah/V']))
                elif statistic == 'log_abs_mean':
                    stat_value = np.log(np.abs(np.mean(Delta_QV_dQdV_dVdQ['Delta_dQdV_Ah/V'])))
                else:
                    raise ValueError(f"Statistic '{statistic}' is not supported.")
                extract_values[index] = stat_value
    return extract_values

def get_delta_DeltaV_SOC_statistic(cell_data: CellData_ISUILCC_NMCGr, measurement_type: str, rate: str, statistic: str) -> list:
    """
    Get a statistic of the delta-DeltaV_SOC curve from week N to week 0 for a given rate.
    """
    extract_tables = cell_data.measurement_extracts[f'delta_DeltaV_SOC_{rate}']
    mask_this_measurement = cell_data.get_measurement_type_mask(measurement_type)
    extract_values = np.ones(len(extract_tables)) * np.nan
    for index, (delta_DeltaV_SOC, is_this_measurement_type) in enumerate(zip(extract_tables, mask_this_measurement)):
        if is_this_measurement_type:
            if cell_data.repeat_num[index] > 1:
                if statistic == 'var':
                    stat_value = np.var(delta_DeltaV_SOC['delta_DeltaV'])
                elif statistic == 'mean':
                    stat_value = np.mean(delta_DeltaV_SOC['delta_DeltaV'])
                elif statistic == 'log_var':
                    stat_value = np.log(np.var(delta_DeltaV_SOC['delta_DeltaV']))
                elif statistic == 'log_mean':
                    stat_value = np.log(np.mean(delta_DeltaV_SOC['delta_DeltaV']))
                else:
                    raise ValueError(f"Statistic '{statistic}' is not supported.")
                extract_values[index] = stat_value
    return extract_values