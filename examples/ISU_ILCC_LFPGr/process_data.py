
# Core
import pickle
import pandas as pd
import numpy as np
# Data
from ISUILCC_LFPGr_Data import Dataset_ISUILCC_LFPGr

# Turn off warnings
import warnings
warnings.filterwarnings('ignore')


# Load data set
data = Dataset_ISUILCC_LFPGr()
# Meta data
meta = data.cell_extracts_and_metadata_to_df()

# Summary data
summary_data = data.get_dataset_summary_table(nan_filter_method='fill_zero')
# Some manual cleaning of this summary data
# Drop columns where all values are 0
summary_data = summary_data.loc[:, (summary_data != 0).any(axis=0)]
# Drop columns that have more than zeros than twice the number of unique cell_ids 
# (many columns have 0 value in first row so we don't want to remove those, just
# columns with mostly crappy data)
num_cells = summary_data['cell_id'].nunique()
summary_data = summary_data.loc[:, (summary_data == 0).sum(axis=0) < 2 * num_cells]
summary_data.drop(columns=['measurement_type', 
                'C_chg', 'C_dis', 'DoD', 'GroupNumber',
                'days_to_80percent_C/3_capacity', 'log_days_to_80percent_C/3_capacity'], inplace=True)
# Replace any inf values with 0
summary_data.replace([np.inf, -np.inf], 0, inplace=True)

# Other data 
failure_data = data.get_failure_matrices()
rul_data = data.get_rul_matrices()
lookahead_data = data.get_lookahead_matrices()
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

# Write entire dataset to h5
data.to_h5(pd.HDFStore('data_processed/ISUILCC_LFPGr_Data.h5', mode='w'))