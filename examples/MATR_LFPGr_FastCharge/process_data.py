# Core
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
# Data
from MATR_LFPGr_FastCharge_Data import Dataset_MATR_LFPGr

# turn off warnings
import warnings
warnings.filterwarnings('ignore')


data = Dataset_MATR_LFPGr(only_study_1=True)

# Meta data
meta = data.cell_extracts_and_metadata_to_df()

# Summary data
summary_data = data.get_dataset_summary_table(nan_filter_method='fill_zero')
# drop columns where all values are 0
summary_data = summary_data.loc[:, (summary_data != 0).any(axis=0)]
# drop columns that have more than zeros than twice the number of unique cell_ids 
# (many columns have 0 value in first row so we don't want to remove those, just
# columns with mostly crappy data)
num_cells = summary_data['cell_id'].nunique()
summary_data = summary_data.loc[:, (summary_data == 0).sum(axis=0) < 2 * num_cells]
summary_data.drop(columns=['measurement_type', 'batch', 'study',
                'policy_str', 'policy_rate_1', 'policy_rate_2', 'policy_rate_3', 'policy_rate_4', 'nominal_charging_time_minutes',
                'MATR_cycle_life', 'remaining_cycle_life', 'log_remaining_cycle_life'], inplace=True)
# replace any inf values with 0
summary_data.replace([np.inf, -np.inf], 0, inplace=True)
# every 50 cycles
summary_data = summary_data[summary_data['cycle'] % 50 == 0].reset_index(drop=True)

# # Other data 
failure_data = data.get_failure_matrices(nan_filter_method='fill_zero')
num_sets = len(failure_data[1])
failure_data = ([failure_data[0][i] for i in np.arange(39, num_sets, 50)],
                [failure_data[1][i] for i in np.arange(39, num_sets, 50)])

rul_data = data.get_rul_matrices(nan_filter_method='fill_zero')
cell_ids, x, y = rul_data[0], rul_data[1][0], rul_data[1][1]
x = x[x['cycle'] % 50 == 0]
y = y[x.index]
cell_ids = np.array(cell_ids)[x.index].tolist()
x, y = x.reset_index(drop=True), y.reset_index(drop=True)
rul_data = (cell_ids, (x, y))

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

data.to_h5(pd.HDFStore('data_processed/MATR_LFPGr_FastCharge_Data.h5', mode='w'))