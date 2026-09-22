# Core
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
# Data
from MATR_NMCGr_Formation_Data import Dataset_MATR_NMCGr

# turn off warnings
import warnings
warnings.filterwarnings('ignore')

data = Dataset_MATR_NMCGr()

# Meta data
meta = data.cell_extracts_and_metadata_to_df()

# Summary data
summary_data = data.get_dataset_summary_table(nan_filter_method='fill_zero')
summary_data.drop(columns=['measurement_type',
                            'formation_temperature', 'ocv_time', 'formation_charge_current_1', 
                            'formation_cutoff_voltage_1', 'formation_charge_current_2', 'formation_verification_repeat'], 
                        inplace=True)

# drop columns where all values are 0
summary_data = summary_data.loc[:, (summary_data != 0).any(axis=0)]
# drop columns that have more than zeros than twice the number of unique cell_ids 
# (many columns have 0 value in first row so we don't want to remove those, just
# columns with mostly crappy data)
num_cells = summary_data['cell_id'].nunique()
summary_data = summary_data.loc[:, (summary_data == 0).sum(axis=0) < 2 * num_cells]
# replace any inf values with 0
summary_data.replace([np.inf, -np.inf], 0, inplace=True)

# # Other data 
failure_data = data.get_failure_matrices(nan_filter_method='fill_zero')
rul_data = data.get_rul_matrices(nan_filter_method='fill_zero')
traj_data = data.get_trajectory_matrices()

with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
with open('summary_data.pkl', 'wb') as f:
    pickle.dump(summary_data, f)
with open('failure_data.pkl', 'wb') as f:
    pickle.dump(failure_data, f)
with open('rul_data.pkl', 'wb') as f:
    pickle.dump(rul_data, f)
with open('traj_data.pkl', 'wb') as f:
    pickle.dump(traj_data, f)

data.to_h5(pd.HDFStore('data_processed/MATR_NMCGr_FastCharge_Data.h5', mode='w'))