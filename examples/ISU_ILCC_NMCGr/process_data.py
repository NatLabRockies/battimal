# Core
import os
import pickle
import pandas as pd
import numpy as np
# Data
from ISUILCC_NMCGr_Data import Dataset_ISUILCC_NMCGr

# turn off warnings
import warnings
warnings.filterwarnings('ignore')

dataset = Dataset_ISUILCC_NMCGr()
# Meta data
meta = dataset.cell_extracts_and_metadata_to_df()

# Summary data
summary_data = dataset.get_dataset_summary_table(nan_filter_method='fill_zero')
summary_data = summary_data[summary_data['measurement_type'] == 'RPT C_5'].reset_index(drop=True)
# drop columns where all values are 0
summary_data = summary_data.loc[:, (summary_data != 0).any(axis=0)]
# drop columns that have more than zeros than twice the number of unique cell_ids 
# (many columns have 0 value in first row so we don't want to remove those, just
# columns with mostly crappy data)
num_cells = summary_data['cell_id'].nunique()
summary_data = summary_data.loc[:, (summary_data == 0).sum(axis=0) < 2 * num_cells]
summary_data.drop(columns=['measurement_type', 
                'mean_C_chg', 'mean_C_dis', 'DoD', 'Stress_charge', 'Stress_discharge', 'Stress_avg', 
                'GroupNumber', 
                #  'remaining_weeks_to_80percent_C_5_capacity', 'log_remaining_weeks_to_80percent_C_5_capacity',
                'weeks_to_80percent_C_5_capacity', 'log_weeks_to_80percent_C_5_capacity'], inplace=True)
# replace any inf values with 0
summary_data.replace([np.inf, -np.inf], 0, inplace=True)

# Other data 
failure_data = dataset.get_failure_matrices()
rul_data = dataset.get_rul_matrices()
traj_data = dataset.get_trajectory_matrices()

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

dataset.to_h5(pd.HDFStore('data_processed/ISU_ILCC_NMCGr.h5', mode='w'))