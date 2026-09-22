
"""
Classes for keeping track of data from:
- Individual electrochemical measurements (EchemMeasurement)
- A sequence of measurements from a single cell (CellData)
- A dataset of test results from multiple cells (BatteryDataset)

The way these classes handle battery data is intended to simplify the process of handling
the many 'dimensions' of battery data required for extracting features and targets for
machine learning, plotting, and comparing measured quantities to metadata or experimental
variables. The key 'dimensions' of battery data are described below:
- (N) cell-wise: a dataset may contain multiple cells. Things like experimental variables 
  (impurity level for a cell material) and some calculated extracts (end-of-life capacity 
  or time to failure) are cell-wise, meaning that pulling these extracts from the dataset 
  will yield a vector of 1:N values for the entire dataset.
- (T) chronologically: for each cell, different measurements are performed in chronological
    order. Note that the actual variables used to denote chronology are varied: time since
    beginning of test, cycle count, repeat count for a specific measurement type, etc.
    Depending on experimental design, every cell may have the same or a different number of
    measurements, and there are often many different 'types' of measurements, so we need to
    be very flexible about this. For example, extracting all the measurements from all cells
    in a dataset would yield N lists of T_n measurements, where T_n is the number of
    measurements for cell n=1:N.
- (M) extract-wise: for each measurement, there are many different variables that can be 
    extracted from the measurement, like capacity, resistance, etc. These are the 'features' 
    and 'targets' used in machine learning models. These are referred to as 'extracts' so
    that we only have a single term. For example, pulling all extracts of all measurements
    of all cells in a dataset would yield N lists of T_n dictionaries of M_t lists, where 
    M_t is the number of extracts for measurement t=1:T_n of cell n=1:N.
Basically, for an entire dataset, there are N cells, each with it's own unique set of T ,
measurements with each measurement having it's own set of M extracts (which may be different 
for different types of measurements conducted on the cell).

So, these classes are layered to impose this structure, keeping terminology and structure
consistent so that once we load a specific dataset, we can easily extract features and targets
for training various models (which often pull 'slices' from different dimensions of the dataset)
and quickly compare modeling results across many datasets. The classes are layered as such:
- BatteryDataset: This is the top-level class that contains a list of CellData objects and any
  metadata that applies to the entire dataset. It contains methods that return extracts from
  the entire dataset, organized in ways that are amenable for training various models.
    - CellData: This class contains a list of EchemMeasurement objects and any metadata that 
      applies to an entire cell. It also contains methods for extracting features and targets 
      from both the entire cell's data (like time to failure) and from individual measurements
      (certain extracts require context from the entire cell's life to calculate, for example,
      relative discharge capacity). The chronological attributes of measurements are stored as
      CellData attributes.
        - EchemMeasurement: This class contains the raw data from a single electrochemical
          measurement, and any methods to calculate extracts from that specific measurement.

Including attributes, the structure of an example dataset is, at minimum:
- Dataset
    - metadata: dict of dataset-wide metadata
    - battery_data: list of CellData objects
        - [
           CellData,
              - cell_metadata: dict of cell metadata (ALWAYS has a 'cell_id')
              - measurements: list of EchemMeasurement objects
                - [
                    EchemMeasurement,
                      - measurement_type: str (type of measurement, like 'formation' or 'RPT')
                      - Current_A: list of current data in amperes for the measurement
                      - Voltage_V: list of voltage data in volts for the measurement
                      - Step: list of step numbers for the measurement
                      - Time_s: list of time data in seconds for the measurement
                      - extractors: list of tuples, where each tuple is the name of an extracted
                          quantity and a function that does the extracting from the measurement
                      - extracts: dict of extracted quantities from the measurement
                    EchemMeasurement,
                    EchemMeasurement,
                    ...
                  ]
              - measurement_type: list of measurement types for each measurement
              - start_time_s: list of start times in seconds since beginning of test for each 
                  measurement
              - start_time_day: list of start times in days since beginning of test for each
                  measurement
              - cycle: list of cumulative cycle numbers for each measurement
              - charge_throughput_Ah: list of cumulative charge throughput in amp-hours for
                  each measurement
              - repeat_num: list of repetition numbers for each measurement
              - measurement_extractors: dict of functions for extracting quantities from the
                  measurements
              - measurement_extracts: dict of extracted quantities from the measurements
              - cell_extractors: dict of functions for extracting quantities from the CellData
                   object
              - cell_extracts: dict of extracted quantities from the CellData object
              - characterization_data: dict of non-electrochemical characterization data from
                  the cell
            CellData,
            CellData,
            ...
          ]
Superclasses implemented from these parent classes tailor these classes to specific datasets, defining how
raw electrochemical data is parsed into EchemMeasurement objects, how/what extracts are calculated, and how the
dataset is saved to and loaded from an HDF5 file.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid

@dataclass
class EchemMeasurement:
    """
    Class for keeping track of data from an electrochemical measurement.
    'IVSt' refers the the key variables in electrochemical data: current (I), voltage (V), step number (S), and 
    time (t). Optionally, other outputs from a battery test can be included, like power, charge throughput, ...
    This class also requires the type of measurement being stored, like 'formation' or 'RPT'. It is intended that 
    for any specific measurement type, there will be a subclass that inherits this to define specific attributes 
    and methods, such as unique plotting, or parsers to read the data in from a raw data file. Parsers could also
    be a defined function that outputs several EchemMeasurement objects, one for each measurement in the raw file.
    
    Each type of data/project/modeling effort has its own needs for extracting data from a specific type of measurement.
    The data class will have a list of extractors, where each entry is the name of an extracted quantity and a function  
    that does the extracting from the RPT. These are assumed to be run in order if certain extracts rely on one another.
    Calculated values are stored in the extracts dict with the specified name.

    :param measurement_type: The type of measurement being stored, like 'formation' or 'RPT'.
    :param Current_A: The current data from the measurement.
    :param Voltage_V: The voltage data from the measurement.
    :param Step: The step number data from the measurement.
    :param Time_s: The time data from the measurement.
    :param extractors: A list of tuples, where each tuple is the name of an extracted quantity and a function
        that does the extracting from the measurement.
    :param extracts: A dictionary of extracted quantities from the measurement.
    """
    # metadata for this measurement
    measurement_type: str
    # data from the measurement
    Current_A: list[float]
    Voltage_V: list[float]
    Step: list[int]
    Time_s: list[float]
    # attribute extractors
    extractors: dict = None
    extracts: dict = None

    def to_df(self) -> pd.DataFrame:
        """
        Convert the EchemMeasurement data to a pandas DataFrame.
        :return: A pandas DataFrame with the measurement data.
        """
        return pd.DataFrame({
            'Current_A': self.Current_A,
            'Voltage_V': self.Voltage_V,
            'Step': self.Step,
            'Time_s': self.Time_s
        })
    
    def to_h5(self, hdf5: pd.HDFStore, measurement_key: str = 'measurement'):
        """
        Save the EchemMeasurement data to an HDF5 file.
        :param hdf5: The HDFStore object to save the data to.
        :param measurement_key: The key for the measurement data in the HDF5 file.
        """
        measurement_df = self.to_df()
        hdf5.put(measurement_key, measurement_df)

    def scalar_extracts_to_df(self) -> pd.DataFrame:
        """
        Return scalar extracts as a DataFrame.
        :return: A pandas DataFrame with the extracts data.
        """
        # Make a dataframe of the extracts that are not dataframes and save them.
        scalar_extracts = {}
        for extract, values in self.extracts.items():
            if is_list_scalar(values):
                scalar_extracts[extract] = values
        return pd.DataFrame.from_dict(scalar_extracts)

    @staticmethod
    def from_h5(measurement_type: str, hdf5: pd.HDFStore, measurement_key: str, ignore_measurement_data: bool = False):
        """
        Load the EchemMeasurement data from an HDF5 file.
        :param measurement_type: The type of measurement being loaded.
        :param hdf5: The HDFStore object to load the data from.
        :param measurement_key: The key for the measurement data in the HDF5 file.
        :param ignore_measurement_data: If True, do not load the measurement data.
        :return: An EchemMeasurement object with the loaded data.
        """
        if not ignore_measurement_data:
            measurement_data = hdf5[measurement_key]
            return EchemMeasurement(measurement_type=measurement_type, 
                                    Current_A=measurement_data['Current_A'].to_list(),
                                    Voltage_V=measurement_data['Voltage_V'].to_list(),
                                    Step=measurement_data['Step'].to_list(),
                                    Time_s=measurement_data['Time_s'].to_numpy(),
                                    extractors=[],
                                    extracts={}
                                    )
        else:
            return EchemMeasurement(measurement_type=measurement_type, 
                                    Current_A=[],
                                    Voltage_V=[],
                                    Step=[],
                                    Time_s=[],
                                    extractors=[],
                                    extracts={}
                                    )
    
    def run_extractors(self):
        """
        Run the extractors for this measurement. This will call each extractor function and store the result in self.extracts.
        """
        if self.extractors is not None:
            if self.extracts is None:
                self.extracts = {}
            # print(f'Measurement level measurement extractor {i}')
            for extract, extractor_func in self.extractors.items():
                # Check type of extract, extractor_func
                if not isinstance(extract, str):
                    raise TypeError(f"Expected string for extract name, got {type(extract)}")
                if not callable(extractor_func):
                    raise TypeError(f"Expected callable for extractor function, got {type(extractor_func)} for extract '{extract}'")
                self.extracts[extract] = extractor_func(self)

class CellData:
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

    Methods:
    - append: Append a new measurement to the cell data. Append to chronological attributes automatically.
    - measurement_metadata_to_df: Return measurement metadata as a DataFrame.
    - scalar_measurement_extracts_to_df: Return scalar measurement extracts as a DataFrame.
    - to_h5: Save the CellData to an HDF5 file.
    - run_extractors: Run the extractors for this cell. This will call each extractor function both at the measurement level
        and cell level.
    - __repr__: Return a string representation of the CellData object.
    - __iter__: Return an iterator through the measurements in CellData.
    - __getitem__: Get a measurement from CellData.measurements by index.
    """
    cell_id: str
    cell_metadata: dict
    characterization_data: dict
    measurements: list[EchemMeasurement]
    measurement_type: list
    measurement__class__: list[str]
    start_time_s: list[float]
    start_time_day: list[float]
    cycle: list[int]
    charge_throughput_Ah: list[float]
    repeat_num: list[int]
    measurement_extractors: dict
    measurement_extracts: dict
    cell_extractors: dict
    cell_extracts: dict

    def __init__(self, cell_id, cell_metadata: dict = None, characterization_data: dict = None, measurement_extractors: dict = None, cell_extractors: dict = None):
        """
        Initialize the CellData object with a cell ID.

        :param cell_metadata: cell metadata, which is the cell ID, this class name, and any other input metadata
        :param characterization_data: characterization data, which is a dictionary of non-electrochemical characterization data from the cell
        :param measurement_extractors: dict of functions for extracting quantities from the measurements. This is intended
            for calculating quantities that require comparison between multiple measurements, for example, capacity loss since
            beginning of life, or delta-Q(V) features. Extracted quantities are stored in the extracts dict in each 
            EchemMeasurement object in the list of measurements. If extracted quantities are specific to a certain type of
            EchemMeasurement, the extractor function should handle that. These functions should always input the CellData object
            and the name of the extract to be calculated, and return the CellData object, modifying the extracts in each
            measurement but not changing the length of measurements.
        :param cell_extractors: dict of functions for extracting quantities from the CellData object.
        """
        # Cell-level metadata
        self.cell_id = cell_id
        self.cell_metadata = {'cell_id': cell_id,
                              '__class__': self.__class__.__name__}
        if cell_metadata is not None:
            self.cell_metadata.update(cell_metadata)
        # Non-electrochemical characterization data
        self.characterization_data = characterization_data
        # Electrochemical measurements: raw data and properties stored in chronological order
        self.measurements = []          # container for list of EchemMeasurement objects for this cell
        self.measurement_type = []      # type of measurement being stored, like 'formation' or 'RPT'
        self.measurement__class__ = []     # class of measurement being stored
        self.start_time_s = []          # start time of each measurement in seconds since beginning of test
        self.start_time_day = []        # start time of each measurement in days since beginning of test
        self.cycle = []                 # cumulative cycle number of each measurement
        self.charge_throughput_Ah = []  # cumulative charge throughput in amp-hours
        self.repeat_num = []            # repetition number of each measurement according to its type
        # Extractors and extracts at measurement-level and cell-level.
        self.measurement_extractors = measurement_extractors  # dict of functions for extracting data from the measurements
        self.measurement_extracts = {}  # dict of extracted data from the measurements.
        self.cell_extractors = cell_extractors    # dict of functions for extracting data from the CellData object.
        self.cell_extracts = {}              # dict of extracted data from the CellData object.

    def append(self, data: EchemMeasurement, start_time_s: float = 0, cycles: int = 0, repeat_modifier: int = 0, charge_throughput_Ah: float = None):
        """
        Append a new measurement to the cell data. Append to chronological attributes automatically.
        Number of cycles in the measurement to be appended is required, as I don't want to try and count them
        using some sort of algorithm, as this is prone to error. Same for time since start of test.

        :param data: The EchemMeasurement object to append.
        :param start_time_s: The start time of the measurement in seconds since beginning of test.
        :param cycles: The number of cycles in the measurement to be appended.
        :param repeat_modifier: An additive modifier to the repeat_num assigned to this measurement (to denote missing data, for example).
        :param charge_throughput_Ah: The cumulative charge throughput in amp-hours since beginning of test.
        """
        self.measurements.append(data)
        self.measurement_type.append(data.measurement_type)
        self.measurement__class__.append(data.__class__.__name__)
        self.start_time_s.append(start_time_s)
        self.start_time_day.append(start_time_s / (24 * 3600))
        self.cycle.append(self.cycle[-1] + cycles if len(self.cycle) > 0 else cycles)
        if charge_throughput_Ah is None:
            charge_throughput_Ah = trapezoid(data.Current_A, np.array(data.Time_s) - data.Time_s[0])/3600

        self.charge_throughput_Ah.append(self.charge_throughput_Ah[-1] + charge_throughput_Ah if len(self.charge_throughput_Ah) > 0 else charge_throughput_Ah)
        # Determine the repeat number for this type of measurements
        repeat_count = self.measurement_type.count(data.measurement_type) - 1
        self.repeat_num.append(repeat_count + 1 + repeat_modifier if repeat_count > 0 else 1 + repeat_modifier)

    def measurement_metadata_to_df(self) -> pd.DataFrame:
        """
        Return measurement metadata as a DataFrame.
        :return: A pandas DataFrame with the measurement metadata.
        """
        expected_length = len(self.measurements)
        # For every attribute in self that is a list of the expected length, store it in a dict
        metadata_dict = {}
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, list) and len(attr_value) == expected_length and attr_name != 'measurements':
                metadata_dict[attr_name] = attr_value
        return pd.DataFrame.from_dict(metadata_dict)
    
    def scalar_measurement_extracts_to_df(self) -> pd.DataFrame:
        """
        Return scalar measurement extracts as a DataFrame.
        :return: A pandas DataFrame with the extracts data.
        """
        # Make a dataframe of the measurement_extracts that are not dataframes and save them.
        scalar_extracts = {}
        for extract, values in self.measurement_extracts.items():
            if is_list_scalar(values):
                scalar_extracts[extract] = values
        return pd.DataFrame.from_dict(scalar_extracts)
    
    def get_cell_summary_table(self) -> pd.DataFrame:
        """
        Return the combined measurement extracts and metadata dataframe
        :return: A pandas DataFrame with the extracts data and metadata.
        """
        cell_summary = self.scalar_measurement_extracts_to_df()
        return cell_summary.join(self.measurement_metadata_to_df().drop(columns='measurement__class__'))
    
    def get_extract_array_by_measurement_type(self, extract: str, measurement_type: str):
        """
        Get an array of extract values for a given measurement type the same length as the original array.
        Values associated with any other measurement type are set to NAN so we can do calculations.
        """
        extract_values = np.array(self.measurement_extracts[extract])
        mask_this_measurement = self.get_measurement_type_mask(measurement_type)
        extract_values[~mask_this_measurement] = np.nan
        return (extract_values, mask_this_measurement)
    
    def get_measurement_type_mask(self, measurement_type: str):
        return np.array(self.measurement_type) == measurement_type
    
    def run_extractors(self):
        """
        Run the extractors for this cell. This will call each extractor function both at the measurement level
        and cell level.
        """
        # Run the measurement extractors that are at the measurement level
        if self.measurements is not None:
            for i, m in enumerate(self.measurements):
                m.run_extractors()
                self.measurements[i] = m
        
        # Pull out extracts from each measurement into the extracts dict of the CellData object. Each extract list should be
        # the same length as the number of measurements, so that each extract can be indexed by measurement index. Since not
        # every measurement will have the same extracts, infill with 'None' so every extract has a valid entry for every measurement.
        # Start by getting all the unique types of extracts from every measurement:
        all_extracts = set()
        for m in self.measurements:
            if m.extracts is not None:
                all_extracts.update(m.extracts.keys())
        # Initialize the extracts dict with empty lists for each extract
        for extract in all_extracts:
            self.measurement_extracts[extract] = [None] * len(self.measurements)
        # Fill the extracts dict with the extracts from each measurement
        for i, m in enumerate(self.measurements):
            if m.extracts is not None:
                for extract, value in m.extracts.items():
                    self.measurement_extracts[extract][i] = value
                # Set m.extracts to None because we copied them to the CellData
                m.extracts = None
                    
        # Run the measurement extractors that are at the cell level
        if self.measurement_extractors is not None:
            for extract, extractor_func in self.measurement_extractors.items():
                # Check type of extract, extractor_func
                if not isinstance(extract, str):
                    raise TypeError(f"Expected string for extract name, got {type(extract)}")
                if not callable(extractor_func):
                    raise TypeError(f"Expected callable for extractor function, got {type(extractor_func)} for extract '{extract}'")
                self.measurement_extracts[extract] = extractor_func(self)

        # Run the cell extractors
        if self.cell_extractors is not None:
            for extract, extractor_func in self.cell_extractors.items():
                self.cell_extracts[extract] = extractor_func(self) 
    
    def to_h5(self, hdf5: pd.HDFStore, prefix="", is_close_hdfstore: bool = True):
        """
        Save the CellData to an HDF5 file.
        """
        # Create a group for this cell
        group_name = str(self.cell_metadata['cell_id'])
        # Create a group for cell metadata
        for key, value in self.cell_metadata.items():
            hdf5.put(f"{prefix}/{group_name}/cell_metadata/{key}", pd.Series([value]))
        # Create a group for characterization data
        if self.characterization_data is not None:
            for key, value in self.characterization_data.items():
                hdf5.put(f"{prefix}/{group_name}/characterization_data/{key}", pd.Series([value]))
        # Save chronological attributes associated with each measurement
        measurement_metadata = self.measurement_metadata_to_df()
        hdf5.put(f"{prefix}/{group_name}/measurement_metadata", measurement_metadata)
        # Save each measurement
        for i, measurement in enumerate(self.measurements):
            measurement_key = f"{prefix}/{group_name}/measurements/measurement_{i}"
            measurement.to_h5(hdf5, measurement_key)
        # Save scalar measurement extracts
        measurement_extracts = self.scalar_measurement_extracts_to_df()
        hdf5.put(f"{prefix}/{group_name}/measurement_extracts/dataframe", measurement_extracts)
        # For measurement extracts that are DataFrame or Series (high dimensional), save them in subgroups
        for extract, values in self.measurement_extracts.items():
            if not is_list_scalar(values):
                for i, value in enumerate(values):
                    if value is not None:
                        hdf5.put(f"{prefix}/{group_name}/measurement_extracts/{extract}/value_{i}", value)
        # Make a dataframe of the cell_extracts and store it
        hdf5.put(f"{prefix}/{group_name}/cell_extracts", pd.DataFrame.from_dict(self.cell_extracts, orient='index'))

        if is_close_hdfstore:
            hdf5.close()

    @classmethod
    def from_h5(cls, cell_id, hdf5: pd.HDFStore, prefix="", is_close_hdfstore: bool = True, measurement_classes: dict = None, 
                ignore_measurement_data: bool = False,
                ignore_high_dim_measurement_extracts: bool = False):
        """
        Load the CellData from an HDF5 file.
        """
        keys = hdf5.keys()
        cell_data = cls(cell_id)
        # Cell metadata
        cell_metadata_keys = [key for key in keys if key.startswith(f'{prefix}/{cell_id}/cell_metadata/')]
        for key in cell_metadata_keys:
            metadata_key = key.split('/')[-1]
            cell_data.cell_metadata[metadata_key] = hdf5[key][0]
        # Measurements and measurement metadata
        measurement_metadata = hdf5[f'{prefix}/{cell_id}/measurement_metadata']
        measurement_keys = [key for key in keys if key.startswith(f'{prefix}/{cell_id}/measurements/')]
        measurement_keys = sorted(measurement_keys, key=lambda x: int(x.split('/')[-1].split('_')[-1]))
        for i, measurement_key in enumerate(measurement_keys):
            measurement__class__ = measurement_metadata.loc[i, 'measurement__class__']
            measurement_type = measurement_metadata.loc[i, 'measurement_type']
            if measurement_classes is not None and measurement__class__ in measurement_classes:
                MeasurementClass = measurement_classes[measurement__class__]
            elif measurement__class__ in globals():
                MeasurementClass = globals()[measurement__class__]
            else:
                raise ValueError(f"Measurement class '{measurement__class__}' not found in 'measurement_classes' arg or globals().")
            assert issubclass(MeasurementClass, EchemMeasurement), f"Expected EchemMeasurement subclass from {MeasurementClass}"
            measurement_data = MeasurementClass.from_h5(measurement_type, hdf5, measurement_key, ignore_measurement_data=ignore_measurement_data)
            cell_data.append(measurement_data)
        for attr in measurement_metadata.columns:
            setattr(cell_data, attr, measurement_metadata[attr].to_list())
        # Scalar measuerement extracts dataframe
        scalar_measurement_extracts = hdf5[f'{prefix}/{cell_id}/measurement_extracts/dataframe']
        for extract in scalar_measurement_extracts.columns:
            cell_data.measurement_extracts[extract] = scalar_measurement_extracts[extract].to_list()
        # High-dimensional measurement extracts
        highdim_measurement_extracts = np.unique([key.split('/')[-2] for key in keys if key.startswith(f'{prefix}/{cell_id}/measurement_extracts/') and not 'dataframe' in key])
        for extract in highdim_measurement_extracts:
            if not ignore_high_dim_measurement_extracts:
                extract_values = []
                value_keys = [key for key in keys if key.startswith(f'{prefix}/{cell_id}/measurement_extracts/{extract}/value_')]
                value_keys = sorted(value_keys, key=lambda x: int(x.split('/')[-1].split('_')[-1]))
                for value_key in value_keys:
                    extract_values.append(hdf5[value_key])
                cell_data.measurement_extracts[extract] = extract_values
            else:
                cell_data.measurement_extracts[extract] = [None] * len(cell_data.measurements)
        # Cell extracts index-wise dataframe
        cell_extracts = hdf5[f'{prefix}/{cell_id}/cell_extracts']
        for extract in cell_extracts.index:
            cell_data.cell_extracts[extract] = cell_extracts.loc[extract][0]

        if is_close_hdfstore:
            hdf5.close()
        return cell_data

    def __repr__(self) -> str:
        """
        Return a string representation of the CellData object.
        The string representation is the cell ID, total number of measurements, and number of each type of measurement.
        There is also a string representing the measurements in order, using a different symbol for each measurement type,
        with the symbol denoted in the previous line. For contiguous repetitions of the same measurement type, the number
        of repetitions is shown in parentheses instead of repeating the symbol many times.
        """
        # Count the number of each type of measurement
        type_count = {t: self.measurement_type.count(t) for t in set(self.measurement_type)}
        # Create a string representation of the measurements in order
        measurement_str = ""
        for i, m in enumerate(self.measurements):
            if i % 50 == 0:
                measurement_str += "\n"
            if i == 0:
                measurement_str += f"'{m.measurement_type}'"
            elif m.measurement_type != self.measurements[i-1].measurement_type:
                measurement_str += f" '{m.measurement_type}'"
            else:
                if self.repeat_num[i] > 1:
                    try:
                        if self.measurements[i+1].measurement_type != m.measurement_type:
                            measurement_str += f"(x{self.repeat_num[i]})"
                    except IndexError:
                        measurement_str += f"(x{self.repeat_num[i]})"
        return f"Cell ID: {self.cell_metadata['cell_id']}, Total measurements: {len(self.measurements)}, Measurement types: {type_count}{measurement_str}"

    def __iter__(self):
        """
        Return an iterator through the measurements in CellData.
        """
        return iter(self.measurements)

    def __getitem__(self, index):
        """
        Get a measurement by index.
        :param index: The index of the measurement to get.
        :return: The EchemMeasurement object at the specified index.
        """
        return self.measurements[index]
    
    def __len__(self):
        """
        Return the number of measurements in the CellData object.
        """
        return len(self.measurements)

    
class BatteryDataset:
    """
    Superclass to define how a battery dataset is stored and how extracts (features and 
    targets) and metadata are exported for various models, so that anything on top of this like
    like modeling, automated experiment planning, and plotting tools can work without needing
    to know the details of any specific dataset.

    Attributes:
    - battery_data: a list of CellData instances.
    - metadata: a dictionary of metadata for the entire dataset, can be useful for cluttering up cell_metadata with
      tons of repeated information.

    Methods:
    - cell_extracts_to_df: convert the cell extracts to a pandas DataFrame.
    - get_earlylife_matrices: get X and Y matrices for training early-life failure prediction models.
    - get_rul_matrices: get X and Y matrices for training remaining useful life (RUL) prediction models
    - get_traj_matrices: get X and Y matrices for training trajectory prediction models
    - get_lookahead_matrices: get X and Y matrices for training near-term look-ahead prediction models
    - get_traj_param_matrices: get X and Y matrices for training trajectory parameter prediction models

    """
    metadata: dict = None
    battery_data: list[CellData]

    def __init__(self, battery_data, metadata=None):
        """
        Initialize the BatteryData class.

        :param metadata: Dataset-wide metadata.
        :param battery_data: A list of CellData instances.
        """
        self.metadata = metadata
        self.battery_data = battery_data
        for cell in self.battery_data:
            if not isinstance(cell, CellData):
                raise TypeError(f"Expected CellData object, got {type(cell)}")
    
    def to_h5(self, hdf5: pd.HDFStore):
        """
        Save the BatteryDataset to an HDF5 file.
        :param hdf5: The HDFStore object to save the data to.
        """
        # Save metadata
        if self.metadata is not None:
            for key, value in self.metadata.items():
                hdf5.put(f'/metadata/{key}', pd.Series([value]))
        # Save each cell's data
        for cell in self.battery_data:
            cell.to_h5(hdf5, prefix='/battery_data', is_close_hdfstore=False)
        hdf5.close()
            
    @staticmethod
    def from_h5(cls, hdf5: pd.HDFStore, celldata_classes: dict = None, measurement_classes: dict = None,
                ignore_measurement_data: bool = False,
                ignore_high_dim_measurement_extracts: bool = False):
        """
        Load the BatteryDataset from an HDF5 file.
        :param hdf5: The HDFStore object to load the data from.
        """
        dataset = []
        keys = hdf5.keys()

        # Cell data
        cell_ids = np.unique([key.split('/')[2] for key in keys if '/metadata/' not in key])        
        for cell_id in cell_ids:
            # Read cell class from cell_metadata
            cell__class__ = hdf5[f'battery_data/{cell_id}/cell_metadata/__class__'][0]
            if celldata_classes is not None and cell__class__ in celldata_classes:
                CellDataClass = celldata_classes[cell__class__]
            elif cell__class__ in globals():
                CellDataClass = globals()[cell__class__]
            else:
                raise ValueError(f"CellData class '{cell__class__}' not found in 'celldata_classes' arg or globals().")
            assert issubclass(CellDataClass, CellData), f"Expected CellData subclass from {CellDataClass}"
            cell_data = CellDataClass.from_h5(cell_id, hdf5, 
                                              prefix='/battery_data', is_close_hdfstore=False, 
                                              measurement_classes=measurement_classes,
                                              ignore_measurement_data=ignore_measurement_data,
                                              ignore_high_dim_measurement_extracts=ignore_high_dim_measurement_extracts)
            dataset.append(cell_data)
        
        # Datset metadata
        metadata = {}
        metadata_keys = [key for key in keys if key.startswith(f'/metadata/')]
        for key in metadata_keys:
            metadata_field = key.split('/')[-1]
            metadata[metadata_field] = hdf5[key][0]
        
        hdf5.close()
        return cls(battery_data=dataset, metadata=metadata)
    
    def run_extractors(self):
        """
        Run the extractors for each cell in the battery dataset.
        This will call each extractor function both at the measurement level and cell level for each cell.
        """
        for cell in self.battery_data:
            cell.run_extractors()

    def cell_extracts_and_metadata_to_df(self) -> pd.DataFrame:
        """
        Convert the cell extracts and metadata to a pandas DataFrame.
        :return: A pandas DataFrame with the cell extracts data.
        """
        cell_extracts_and_metadata = {}
        for i, cell in enumerate(self.battery_data):
            if i == 0:
                for metadata, value in cell.cell_metadata.items():
                    if is_value_scalar(value):
                        cell_extracts_and_metadata[metadata] = [value]
                    else:
                        cell_extracts_and_metadata[metadata] = [np.nan]
                for extract, value in cell.cell_extracts.items():
                    if is_value_scalar(value):
                        cell_extracts_and_metadata[extract] = [value]
                    else:
                        cell_extracts_and_metadata[extract] = [np.nan]
            else:
                for metadata, value in cell.cell_metadata.items():
                    if is_value_scalar(value):
                        cell_extracts_and_metadata[metadata].append(value)
                    else:
                        cell_extracts_and_metadata[metadata].append(np.nan)
                for extract, value in cell.cell_extracts.items():
                    if is_value_scalar(value):
                        cell_extracts_and_metadata[extract].append(value)
                    else:
                        cell_extracts_and_metadata[extract].append(np.nan)
        return pd.DataFrame.from_dict(cell_extracts_and_metadata)
    
    def get_dataset_summary_table(self, nan_filter_method='none', n_skip_measurements=0) -> pd.DataFrame:
        """
        Return the combined measurement extracts and metadata dataframe for the entire dataset.

        :param nan_filter_method: Method to filter NaN values from the feature and target matrices. Options are 'drop_row' to drop 
            any rows with NaN values, 'drop_column' to drop any columns with NaN values, and 'fill_zero' to fill NaN values with zero.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell. This can be
            useful if certain vars are NaN at beginning of life but become valid later on, for example.
        :return: A pandas DataFrame with the extracts data and metadata.        
        """
        summarydata = None
        for cell in self:
            cell_summary = cell.get_cell_summary_table()
            for metadata, value in cell.cell_metadata.items():
                if metadata == '__class__':
                    continue
                if is_value_scalar(value):
                    cell_summary[metadata] = value
            for extract, value in cell.cell_extracts.items():
                if is_value_scalar(value):
                    cell_summary[extract] = value
            cell_summary = cell_summary[n_skip_measurements:]
            if summarydata is None:
                summarydata = cell_summary
            else:
                summarydata = pd.concat((summarydata, cell_summary), axis=0)
        summarydata.reset_index(drop=True, inplace=True)

        if nan_filter_method == 'drop_column':
            summarydata = summarydata.dropna(axis=1, how='any')
        elif nan_filter_method == 'drop_row':
            summarydata = summarydata.dropna(axis=0, how='any').reset_index(drop=True)
        elif nan_filter_method == 'fill_zero':
            summarydata = summarydata.fillna(0)
        elif nan_filter_method == 'none':
            pass
        else:
            raise ValueError(f"Expected nan_filter_method to be one of 'none', 'drop_row', 'drop_column', or 'fill_zero', got {nan_filter_method}")
        
        return summarydata

    def __iter__(self):
        """
        Return an iterator through the battery data.
        """
        return iter(self.battery_data)
    
    def __getitem__(self, index):
        """
        Get a cell by index.
        :param index: The index of the cell to get.
        :return: The CellData object at the specified index.
        """
        return self.battery_data[index]
    
    def __len__(self):
        """
        Return the number of cells in the battery data.
        """
        return len(self.battery_data)
 
    """
    Methods for returning matrices for training models.
    """
    def get_failure_matrices(self, target_var_name, measurement_type=None, alignment_var=None, 
                             cumulative=False, nan_filter_method='drop_column',
                             n_skip_measurements=0, force_consistent_vars=False):
        """
        Get the X and Y matrices for training failure prediction models.
        If "cumulative" is set to False, for each point in a chronological sequence 1:T,
        like RPT_1, RPT_2, ... RPT_T, return a feature matrix, X, of the shape (N, M),
        where N is the number of cells in the dataset and M is the number 
        of features, and a target vector, Y, of the shape (N, 1), where N is the number of cells in the
        dataset. If "cumulative" is set to True, each feature matrix, X, has shape (N, M*T),
        where N is the number of cells in the dataset, M is the number of features,
        T is the number of steps in the chronological sequence.
        
        For each feature matrix, X, the target vector Y is unchanged, except for when a cell fails
        or is removed from the dataset, in which case there is no target value for that cell anymore, so the
        number of cells (rows) in both X and Y at any given time point, T, may change, denoted N_T.
        Returns a tuple with the list of the chronogical variable values and a list of (X, Y) tuples.
        
        :param target_var_name: The name of the target variable in the cell extracts DataFrame.
        :param measurement_type: The type of measurement to use for the features. If None, will not filter by measurement type.
        :param alignment_var: The name of the variable in the measurement extracts to align the measurements by. The alignment
            variable is assumed to be a value that is uniform across all measurements that 'share' the value, something like the
            cycle, repeat_num, or some other defined variable in measurement_extracts. If the values are not shared across cells,
            then the algorithm will not work. If None, will assume the measurements are already aligned by their measurement index.
        :param cumulative: Whether or not the X feature matrices include all features up to time step T, or only at time step T.
        :param nan_filter_method: Method to filter NaN values from the feature and target matrices. Options are 'drop_row' to drop 
            any rows with NaN values, 'drop_column' to drop any columns with NaN values, and 'fill_zero' to fill NaN values with zero.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell. This can be
            useful if certain vars are NaN at beginning of life but become valid later on, for example.
        :param force_consistent_vars: If True, will only keep variables in X matrices that are consistent across all time points T.
                            
        :return: A tuple of (
            list of cell ID lists for cells that have valid data at each time point T,
            list of (X [N_T, M], Y [N_T, 1]) tuples at each time point T
            )
        """

        if cumulative:
            raise UserWarning("Cumulative features will not work well for failure models if measurements are not aligned by index across cells.")
        
        y = self.cell_extracts_and_metadata_to_df()

        # Get the max number of measurements
        max_no_measurements = 0
        cell_extracts = []
        cell_ids = []
        for cell in self:
            extracts = cell.get_cell_summary_table()
            cell_extracts.append(extracts)
            cell_ids.append(cell.cell_id)
            no_measurements = len(extracts)
            if no_measurements > max_no_measurements:
                max_no_measurements = no_measurements

        xy_per_measurement = []
        cell_ids_per_measurement = []

        for i in range(n_skip_measurements, max_no_measurements):
            cell_id = []
            x = []
            for extracts, id in zip(cell_extracts, cell_ids):
                if i < len(extracts) and (measurement_type is None or extracts['measurement_type'].iloc[i] == measurement_type):
                    if len(x) == 0:
                        x = extracts.iloc[i:i+1, :]
                    else:
                        x = pd.concat((x, extracts.iloc[i:i+1, :]))
                    cell_id.append(id)
            if len(x) == 0:
                continue
            
            x.insert(0, 'cell_id', cell_id)
            y_i = y[y['cell_id'].isin(x['cell_id'])][target_var_name]
            ### TODO ###
            ###### 'cumulative' won't work right if measurements aren't aligned by index across cells
            ###### so this step is out of order, should go after alignment below. But for clean datasets
            ###### this will still work.
            if cumulative and i > 0:
                x = x.rename(columns={col: f"{col}_{i}" for col in x if col != 'cell_id'})
                x = xy_per_measurement[-1][0].merge(x, how='right', on='cell_id')
            # Clean data: reindex, get rid of columns where any values are NaN or Inf, reindex
            x = x.reset_index(drop=True)
            y_i = y_i.reset_index(drop=True)
            for col in x.columns:
                if pd.api.types.is_float_dtype(x[col]):
                    x[col] = x[col].replace([np.inf, -np.inf], np.nan)
            if nan_filter_method == 'drop_column':
                x = x.dropna(axis=1, how='any')
            elif nan_filter_method == 'drop_row':
                x = x.dropna(axis=1, how='all')
                # drop any rows that have nan values in any column, keeping indices to drop from y as well
                x = x.dropna(axis=0, how='any')
                y = y.loc[x.index]
                # index cell_id by x.index
                cell_id = [cell_id[idx] for idx in x.index]
                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)
            elif nan_filter_method == 'fill_zero':
                x = x.fillna(0)
            else:
                raise ValueError(f"Expected nan_filter_method to be one of 'drop_row', 'drop_column', or 'fill_zero', got {nan_filter_method}")

            y_i = pd.DataFrame(y_i)

            xy_per_measurement.append((x, y_i))
            cell_ids_per_measurement.append(cell_id)

        # remove redundant cell_id column from x
        for i in range(len(xy_per_measurement)):
            xy_per_measurement[i][0].drop(columns=['cell_id'], inplace=True)
        
        # sometimes, xy_per_measurement can have entries from the same set of 'measurements' (like, capacity check number N) that are not aligned
        # by the index of the measurements due to some difference between measurement order between cells (cell 1: A B A B A B ...; cell 2: A A B A B A).
        # Manually fix this if specified by the user
        if alignment_var is not None:
            cell_ids_per_measurement, xy_per_measurement = align_xy_pairs_by_variable(cell_ids_per_measurement, xy_per_measurement, alignment_var)
        
        if force_consistent_vars:
            # Find the intersection of all columns in each x matrix
            common_columns = set(xy_per_measurement[0][0].columns)
            for x, y in xy_per_measurement[1:]:
                common_columns = common_columns.intersection(set(x.columns))
            common_columns = list(common_columns)
            # Keep only the common columns in each x matrix
            for i in range(len(xy_per_measurement)):
                xy_per_measurement[i] = (xy_per_measurement[i][0][common_columns], xy_per_measurement[i][1])
        
        return cell_ids_per_measurement, xy_per_measurement
    
    
    def get_rul_matrices(self, target_var_name, measurement_type=None, nan_filter_method='drop_row', n_skip_measurements=0):
        """
        Get the X and Y matrix for training remaining useful life (RUL) prediction model.
        Unlike the failure prediction models, where there is different (X, Y) pair for each
        chronological variable value, the RUL prediction model has a single feature matrix and target vector.
        At any time point 1:T for every cell, the target is the remaining useful life (RUL) of that cell from
        that point. The features at each time point 1:T and RUL at each point 1:T for each cell are stacked
        into a single feature matrix and target vector. There's no generalizable definition of what 'remaining
        useful life' means, so the user needs to precalculate the target variable. The target should vary
        throughout lifetime, so it is expected to be in the output of `scalar_measurement_extracts_to_df()`.

        :param target_var_name: The name of the target variable in the cell extracts DataFrame.
        :param measurement_type: The type of measurement to use for the features. If None, will not filter by measurement type.
        :param nan_filter_method: Method to filter NaN values from the feature and target matrices. Options are 'drop_row' to drop any rows with NaN values,
            'drop_column' to drop any columns with NaN values, and 'fill_zero' to fill NaN values with zero.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell. This can be
            useful if certain vars are NaN at beginning of life but become valid later on, for example.

        :return: A tuple of (
                    list of cell IDs,
                    a tuple of (X [N, M], Y [N, 1]) where N is the total number of measurements across all cells 
                        and M is the number of features.
                )
        """
        x = None
        y = None
        cell_ids = None
        for cell in self:
            extracts = cell.get_cell_summary_table()
            if measurement_type is not None:
                extracts = extracts.loc[extracts['measurement_type'] == measurement_type, :]
            extracts = extracts.iloc[n_skip_measurements:, :]
            targets = extracts.pop(target_var_name)
            mask = ~targets.isna()
            targets = targets[mask]
            extracts = extracts[mask]
            cell_id = [cell.cell_id] * len(extracts)
                       
            if x is None:
                x = extracts
                y = targets
                cell_ids = cell_id
            else:
                x = pd.concat([x, extracts])
                y = pd.concat([y, targets])
                cell_ids.extend(cell_id)

        # Clean data: reindex, get rid of columns where all values are NaN or Inf
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for col in x.columns:
            if pd.api.types.is_float_dtype(x[col]):
                x[col] = x[col].replace([np.inf, -np.inf], np.nan)
        if nan_filter_method == 'drop_column':
            x = x.dropna(axis=1, how='any')
        elif nan_filter_method == 'drop_row':
            x = x.dropna(axis=1, how='all')
            # drop any rows that have nan values in any column, keeping indices to drop from y as well
            x = x.dropna(axis=0, how='any')
            y = y.loc[x.index]
            # index cell_ids by x.index
            cell_ids = [cell_ids[i] for i in x.index]
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
        elif nan_filter_method == 'fill_zero':
            x = x.fillna(0)
        else:
            raise ValueError(f"Expected nan_filter_method to be one of 'drop_row', 'drop_column', or 'fill_zero', got {nan_filter_method}")

        return cell_ids, (x, y)


    def get_lookahead_matrices(self, target_var_name,  measurement_type=None, alignment_var=None, 
                               cumulative=False, nan_filter_method='drop_column',
                               n_skip_measurements=0, force_consistent_vars=False):
        """
        Get the X and Y matrices for training near-term look-ahead prediction models.
        Near-term look-ahead prediction models are essentially the same as failure prediction models, 
        but rather than predicting the time to failure or capacity at end-of-life, the near-term look-ahead 
        prediction model predicts the target at some time point in the near future, T+1, T+2, ... for any given cell,
        if T = T_max, the 'look ahead' model is equivalent to the early-life failure prediction model.
        So, for any given time T, there is a feature matrix of N cells by M features, and a target vector of N cells
        for each cell that has valid data at that time T. This function returns a matrix Y that is N cells by T, as
        this matrix can easily by filtered by the measurement repeat of interest. 
        The feature matrices are indexed by cell_id, as each given time point T may have valid data from a different
        number of total cells.

        :param target_var_name: The name of the target variable in the cell extracts DataFrame.
        :param measurement_type: The type of measurement to use for the features. If None, will not filter by measurement type.
        :param alignment_var: The name of the variable in the measurement extracts to align the measurements by. The alignment
            variable is assumed to be a value that is uniform across all measurements that 'share' the value, something like the
            cycle, repeat_num, or some other defined variable in measurement_extracts. If the values are not shared across cells,
            then the algorithm will not work. If None, will assume the measurements are already aligned by their measurement index.
        :param cumulative: Whether or not the X feature matrices include all features up to time step T, or only at time step T.
        :param nan_filter_method: Method to filter NaN values from the feature and target matrices. Options are 'drop_row' to drop any rows with NaN values,
            'drop_column' to drop any columns with NaN values, and 'fill_zero' to fill NaN values with zero.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell. This can be
            useful if certain vars are NaN at beginning of life but become valid later on, for example.
        :param force_consistent_vars: If True, will only keep variables in X matrices that are consistent across all time points T.
        
        :return: A tuple of (
            list of cell ID lists for cells that have valid data at each time point T,
            list of feature matrices X [N_T, M] at each time point T, target matrix Y [N, T] where N is the total number of cells 
                and T is the max number of measurements
            )
        """
        if cumulative:
            raise UserWarning("Cumulative features will not work well for look-ahead models if measurements are not aligned by index across cells.")

        cell_ids = []
        target_matrix = None
        if alignment_var is not None:
            target_vars = [target_var_name, alignment_var]
        else:
            target_vars = [target_var_name]
        # In this loop:
        # - Grab targets, extracts, and alignment var values
        # - Get the max number of measurements of any cell
        max_no_measurements = 0
        cell_extracts = []
        for cell in self:
            cell_ids.append(cell.cell_id)
            # Extracts
            extracts = cell.get_cell_summary_table()
            if measurement_type is not None:
                extracts = extracts.loc[extracts['measurement_type'] == measurement_type, :]
            extracts = extracts.iloc[n_skip_measurements:, :]
            cell_extracts.append(extracts)
            # Max length of extracts for any cell
            no_measurements = len(extracts)
            if no_measurements > max_no_measurements:
                max_no_measurements = no_measurements
            # Targets
            targets_this_cell = extracts[target_vars]
            targets_this_cell = targets_this_cell.dropna(axis=0, how='any')
            if alignment_var is not None:
                targets_this_cell = targets_this_cell.pivot_table(columns=alignment_var, values=target_var_name).reset_index(drop=True)
            else:
                targets_this_cell = targets_this_cell.transpose().reset_index(drop=True)
            if target_matrix is None:
                target_matrix = targets_this_cell
            else:
                target_matrix = pd.concat([target_matrix, targets_this_cell], axis=0, ignore_index=True)
        target_matrix = target_matrix.reset_index(drop=True)
        # reorder the columns of target_matrix to be in sorted order
        target_matrix = target_matrix.reindex(sorted(target_matrix.columns), axis=1)
        target_matrix.insert(0, 'cell_id', cell_ids)

        # Get extracts and cell_ids from measurement i of each cell into a list of matrices
        cell_ids_per_measurement = []
        x_per_measurement = []
        for i in range(max_no_measurements):
            x = []
            cell_ids_i = []
            for cell_id, extracts in zip(cell_ids, cell_extracts):
                if i < len(extracts):
                    extracts = extracts.iloc[i:i+1, :]
                    if not np.isnan(extracts[target_var_name].iloc[0]):
                        cell_ids_i.append(cell_id)
                        if len(x) == 0:
                            x = extracts
                        else:
                            x = pd.concat((x, extracts))
            if len(x) > 0:
                x.insert(0, 'cell_id', cell_ids_i)
                ### TODO ###
                ###### 'cumulative' won't work right if measurements aren't aligned by index across cells
                if cumulative:
                    x = x.rename(columns={col: f"{col}_{i}" for col in x if col != 'cell_id'})
                    if i > 0:
                        x = x_per_measurement[-1].merge(x, how='right', on='cell_id')
                
                cell_ids_per_measurement.append(cell_ids_i)
                x_per_measurement.append(x)
        
        # sometimes, xy_per_measurement can have entries from the same set of 'measurements' (like, capacity check number N) that are not aligned
        # by the index of the measurements due to some difference between measurement order between cells (cell 1: A B A B A B ...; cell 2: A A B A B A).
        # Manually fix this if specified by the user.
        # For lookahead, do this before getting the y matrix pairs, since the target matrix is built assuming the measurements are aligned.
        xy_per_measurement = [(x, None) for x in x_per_measurement]
        if alignment_var is not None:
            cell_ids_per_measurement, xy_per_measurement = align_xy_pairs_by_variable(cell_ids_per_measurement, xy_per_measurement, alignment_var)
        x_per_measurement = [xy[0] for xy in xy_per_measurement]

        # pair each feature matrix with the target vector for the corresponding measurement
        xy_per_measurement = []
        for i, (cell_ids, x) in enumerate(zip(cell_ids_per_measurement, x_per_measurement)):
            y = target_matrix[target_matrix['cell_id'].isin(x['cell_id'])]
            # Clean data: reindex, get rid of columns where all values are NaN or Inf, rows where any value is NaN or Inf, reindex
            x = x.drop(columns=['cell_id'])
            y = y.drop(columns=['cell_id'])
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
            for col in x.columns:
                if pd.api.types.is_float_dtype(x[col]):
                    x[col] = x[col].replace([np.inf, -np.inf], np.nan)
            if nan_filter_method == 'drop_column':
                x = x.dropna(axis=1, how='any')
            elif nan_filter_method == 'drop_row':
                x = x.dropna(axis=1, how='all')
                # drop any rows that have nan values in any column, keeping indices to drop from y as well
                x = x.dropna(axis=0, how='any')
                y = y.loc[x.index]
                # index cell_ids by x.index
                cell_ids = [cell_ids[idx] for idx in x.index]        
                cell_ids_per_measurement[i] = cell_ids
                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)
            elif nan_filter_method == 'fill_zero':
                x = x.fillna(0)
            else:
                raise ValueError(f"Expected nan_filter_method to be one of 'drop_row', 'drop_column', or 'fill_zero', got {nan_filter_method}")

            # Index y to only 'look ahead' columns from current week i
            if i < len(x_per_measurement) - 1:
                y = y.iloc[:, i+1:]
                y = y.dropna(axis=1, how='all')
                y = y.dropna(axis=0, how='all')
                x = x[x.index.isin(y.index)]
                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)
                xy_per_measurement.append((x, y))

        if force_consistent_vars:
            # Find the intersection of all columns in each x matrix
            common_columns = set(xy_per_measurement[0][0].columns)
            for x, y in xy_per_measurement[1:]:
                common_columns = common_columns.intersection(set(x.columns))
            common_columns = list(common_columns)
            # Keep only the common columns in each x matrix
            for i in range(len(xy_per_measurement)):
                xy_per_measurement[i] = (xy_per_measurement[i][0][common_columns], xy_per_measurement[i][1])

        return cell_ids_per_measurement[1:], xy_per_measurement
    
    def get_autoregressive_matrices(self, target_var_name, measurement_type=None, alignment_var=None,
                                    n_lags=1, step_size=1, is_diff_x=True, is_diff_y=True, 
                                    nan_filter_method='drop_row', n_skip_measurements=0):
        """
        Get the X and Y matrices for training autoregressive prediction models.
        An autoregressive prediction model predicts the value of some target variable at some time point T using
        previously measured values of that variable at time points T-1, T-2, ... and T-n_lags. Here, we also include all other
        extracted values from the dataset at time points T-1, T-2, ... and T-n_lags as features, as well as the chronological
        variable and its lags. The feature and target matrix from all cells are stacked together to form a single feature 
        matrix X and target vector Y. This is very similar to the look-ahead problem, but instead of having a separate model
        for each measurement, the models trains on data from lags of all measurements, so the shape of the (X,Y) data is actually
        more similar to that of the remaining useful life prediction problem, but the features and targets are lagged values.

        :param target_var_name: The name of the target variable in the cell extracts DataFrame.
        :param measurement_type: The type of measurement to use for the features. If None, will not filter by measurement type.
        :param n_lags: The number of lags 1:n_lags to include in the feature matrix.
        :param step_size: The step size between samples of x and y. This can be useful if the data set is extremely large and
            you want to skip samples for speed, or because samples are highly correlated at small step size.
        :param is_diff_x: If True and n_lags > 1, the returned feature matrix is the features at T-1 and the diff of all other features at each lag > 1.
        :param is_diff_y: If True, the returned targets are the n-lags lagged differences of the target variable, otherwise the targets are the raw values.
        :param nan_filter_method: Method to filter NaN values from the feature and target matrices. Options are 'drop_row' to drop any rows with NaN values,
            'drop_column' to drop any columns with NaN values, and 'fill_zero' to fill NaN values with zero.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell. This can be
            useful if certain vars are NaN at beginning of life but become valid later on, for example.

        :return: A tuple of (
                    list of cell IDs,
                    a tuple of (X [N, M], Y [N, max(T)]) where N is the number of measurements at T>n_lags across all cells and M is the number of features.
            )
        """
        if n_lags.__class__ == int:
            n_lags = np.arange(1, n_lags + 1).tolist()
        elif n_lags.__class__ == list:
            assert all([isinstance(n, int) and n > 0 for n in n_lags]), f"Expected list of positive integers for n_lags, got {n_lags}"
        n_lags = n_lags + [0]
        n_lags = sorted(n_lags)

        # Stack cell_id, rows of x, rows of y from each cell into one big matrix
        x = None
        y = None
        cell_ids = []
        for cell in self:
            # Grab x and y tables from this cell
            extracts = cell.get_cell_summary_table()
            if measurement_type is not None:
                extracts = extracts.loc[extracts['measurement_type'] == measurement_type, :]
            extracts = extracts.iloc[n_skip_measurements:, :].reset_index(drop=True)
            targets = pd.DataFrame(extracts[target_var_name])
            if is_diff_x:
                # keep only numeric columns of extracts
                extracts = extracts.select_dtypes(include=[np.number])
            # Stack each row
            for index in range(max(n_lags), len(extracts)-step_size, step_size):
                cell_ids.append(cell.cell_id)
                # get x row, rearranging multiple lags of x into one big row if desired
                indices = sorted([index - n for n in n_lags])
                x_slice = extracts.iloc[indices, :].reset_index(drop=True)
                if max(n_lags) >= 1:
                    # the 'current' row of x is the last one, all previous ones are from the past
                    # old rows of x will be turned into lag features by renaming and then concatenating to make one row
                    x_0 = x_slice.iloc[[-1], :].reset_index(drop=True)
                    if is_diff_x:
                        x_lags = x_slice.diff(periods=-1).iloc[:-1, :].reset_index(drop=True)
                    else:
                        x_lags = x_slice.iloc[:-1, :].iloc[::-1, :].reset_index(drop=True)
                    # for each lagged row, rename its columns to {col}_lag{n} and merge them onto x_minus1 into one single row
                    for lag in range(len(x_lags)):
                        x_lagN = x_lags.iloc[[lag], :].rename(columns={col: f"{col}_lag{n_lags[lag+1]}" for col in x_0.columns}).reset_index(drop=True)
                        x_0 = x_0.merge(x_lagN, how='right', left_index=True, right_index=True)
                    x_slice = x_0
                
                # targets: 
                y_future = targets.iloc[index+step_size:, :].reset_index(drop=True)
                if is_diff_y:
                    y_minus1 = targets.iloc[index, 0]
                    y_future = y_future - y_minus1
                y_future = y_future.transpose()
                
                if x is None:
                    x = x_slice
                    y = y_future
                else:
                    x = pd.concat([x, x_slice])
                    y = pd.concat([y, y_future])
                

        # Clean data: reindex, get rid of columns where all values are NaN or Inf,
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for col in x.columns:
            if pd.api.types.is_float_dtype(x[col]):
                x[col] = x[col].replace([np.inf, -np.inf], np.nan)
        if nan_filter_method == 'drop_column':
            x = x.dropna(axis=1, how='any')
        elif nan_filter_method == 'drop_row':
            x = x.dropna(axis=1, how='all')
            # drop any rows that have nan values in any column, keeping indices to drop from y as well
            x = x.dropna(axis=0, how='any')
            y = y.loc[x.index]
            # index cell_ids by x.index
            cell_ids = [cell_ids[i] for i in x.index]
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
        elif nan_filter_method == 'fill_zero':
            x = x.fillna(0)
        else:
            raise ValueError(f"Expected nan_filter_method to be one of 'drop_row', 'drop_column', or 'fill_zero', got {nan_filter_method}")
        y = y.dropna(axis=1, how='all')
        return cell_ids, (x, y)
    
    
    def get_trajectory_matrices(self, target_var_name, chronological_var_name,  measurement_type=None, n_skip_measurements=0):
        """
        Get the X and Y matrices for training trajectory prediction models.
        Trajectory prediction models predict the trajectory of a cell over some chronological variable, like
        time, cycle number, cumulative charge throughput, etc. Each cell has a different trajectory and thus it's
        own unique trajectory model. These models are assumed to be one dimenionsal, using just the chronological
        variable as input to predict a single target variable, most commonly, capacity as a function of time. So
        each cell, 1:N, will have an X vector 1:T_N and Y vector 1:T_N, where T_N is the number of time points
        for that specific cell N.
        Return a list of dataframe tuples:
        (X[1:T_N], Y[1:T_N]) for all cells.

        :param target_var_name: The name of the target variable in the cell extracts DataFrame.
        :param chronological_var_name: The name of the chronological variable in the cell extracts DataFrame.
        :param measurement_type: The type of measurement to use for the features. If None, will not filter by measurement type.
        :param n_skip_measurements: Number of initial measurements to skip in the chronological sequence for each cell.

        :return: A tuple of (
                    list of cell IDs, 
                    list of (X [1:T_N], Y [1:T_N]) tuples for each cell
                )
        """
        assert chronological_var_name in ['cycle', 'repeat_num', 'start_time_s', 'start_time_day', 'charge_throughput_Ah'], \
            f"Expected chronological_var_name to be one of 'cycle', 'repeat_num', 'start_time_s', 'start_time_day', or 'charge_throughput_Ah', got {chronological_var_name}"

        xy_per_cell = []
        cell_ids = []
        for cell in self:
            cell_ids.append(cell.cell_id)
            x = np.array(cell.__getattribute__(chronological_var_name))
            if measurement_type is not None:
                x = x[pd.Series(cell.measurement_type) == measurement_type]
            x = x[n_skip_measurements:]
            y = np.array(cell.measurement_extracts[target_var_name])
            if measurement_type is not None:
                y = y[pd.Series(cell.measurement_type) == measurement_type]
            y = y[n_skip_measurements:]
            mask = ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            xy_per_cell.append((pd.DataFrame({chronological_var_name: x}), pd.DataFrame({target_var_name: y})))
        return cell_ids, xy_per_cell        
    

def align_xy_pairs_by_variable(cell_ids: list, xy_pairs: list, alignment_var: str, is_ordered: bool = True) -> tuple:
    """
    Align a list of (X, Y) tuples and lists of cell_id lists by a specified variable in the X DataFrame.
    The alignment variable is assumed to be a value that is uniform across all measurements that 'share' the value, something like the
    cycle, repeat_num, or some other defined variable in measurement_extracts. If the values are not shared across cells,
    then the algorithm will not work. If None, will assume the measurements are already aligned by their measurement index.

    :param cell_ids: A list of cell ID lists for each (X, Y) tuple.
    :param xy_pairs: A list of (X, Y) tuples to align.
    :param alignment_var: The name of the variable in the X DataFrame to align the measurements by.
    :param is_ordered: If True, assume the alignment variable values are ordered and contiguous, like cycle number or repeat_num.
    :return: A tuple of (
                list of cell ID lists for cells that have valid data at each time point T,
                list of (X [N_T, M], Y [N_T, 1]) tuples at each time point T
            )
    """
    # sometimes, xy_per_measurement can have entries from the same set of 'measurements' (like, capacity check number N) that are not aligned
    # by the index of the measurements due to some difference between measurement order between cells (cell 1: A B A B A B ...; cell 2: A A B A B A).
    # Manually fix this if specified by the user
    alignment_var_values = []
    for (x, y) in xy_pairs:
        alignment_var_values.extend(x[alignment_var].unique().tolist())
    alignment_var_values, indices = np.unique(alignment_var_values, return_index=True)
    if not is_ordered:
        alignment_var_values = alignment_var_values[np.sort(indices)]
    num_alignment_var_values = len(alignment_var_values)
    valid_cell_ids = [None] * (num_alignment_var_values)
    valid_xy_pairs = [None] * (num_alignment_var_values)
    for n_var, alignment_val in enumerate(alignment_var_values):
        cell_ids_n = []
        x_n = []
        y_n = []
        for cell_id, (x, y) in zip(cell_ids, xy_pairs):
            if np.any(x[alignment_var] == alignment_val):
                idx = np.argwhere(x[alignment_var] == alignment_val).flatten()
                for idx_row in idx:
                    cell_ids_n.append(cell_id[idx_row])
                x_n.append(x.iloc[idx])
                if y is not None:
                    y_n.append(y.iloc[idx])
                else:
                    y_n.append(pd.Series(None))
        if len(cell_ids_n) > 0:
            valid_cell_ids[n_var] = cell_ids_n
            valid_xy_pairs[n_var] = (pd.concat(x_n, ignore_index=True).reset_index(drop=True),
                                    pd.concat(y_n, ignore_index=True).reset_index(drop=True),)
    return valid_cell_ids, valid_xy_pairs

def is_list_scalar(input: list) -> bool:
    """
    Check all the values in the input list are scalar (not a list, np array, DataFrame, or Series).
    :param extract: The name of the extract to check.
    :return: True if the extract is scalar, False otherwise.
    """
    is_scalar = True
    for val in input: 
        if not is_value_scalar(val):
            is_scalar = False
            break
    return is_scalar

def is_value_scalar(input) -> bool:
    """
    Check if the input is a scalar value (not a list, np array, DataFrame, or Series).
    :param input: The input to check.
    :return: True if the input is scalar, False otherwise.
    """
    return not (isinstance(input, list) or isinstance(input, np.ndarray) or isinstance(input, pd.DataFrame) or isinstance(input, pd.Series))