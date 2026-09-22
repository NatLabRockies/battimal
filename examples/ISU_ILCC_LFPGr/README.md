# ISU-ILCC LFP-Gr Data set
Paul Gasper, NLR, 2026

Download raw data from https://digitalcommons.lib.uconn.edu/reil_datasets/1/. The original authors provide some example data processing code at https://github.com/REIL-UConn/rapid-soh-estimation-from-short-pulses, some of which is copied here. Uncompressed raw data files and subfolders should then all be placed in `data_raw` folder, for example, there should be a `data_raw/CyclingParameters.csv` file and `data_raw/rpt_data` folder.

`battimal` data processing code for this data set is defined in `ISUILCC_LFPGr_Data.py`. Example Jupyter notebooks show the structure of the processed data on a subset of the cells. The script `process_data.py` reads in the raw data, exports matrices structured for training various model types, and saves the processed data as an .h5 data base.
