# MATR LFP-Gr Fast-charge Data set
Paul Gasper, NLR, 2026

Well known as the Severson data set, Stanford data set, D3Batt data set, ...

Download raw data from https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 and https://data.matr.io/1/projects/5d80e633f405260001c0b60a. The original authors provide some example data processing code at https://doi.org/10.5281/zenodo.10648587, some of which is copied here. Uncompressed raw data files and subfolders should then all be placed in `data_raw` folder, for example, there should be several `.mat` files with file names like `2017-05-12_batchdata_updated_struct_errorcorrect.mat`.

`battimal` data processing code for this data set is defined in `MATR_LFPGr_FastCharge_Data.py`. Example Jupyter notebooks show the structure of the processed data on a subset of the cells. The script `process_data.py` reads in the raw data, exports matrices structured for training various model types, and saves the processed data as an .h5 data base.
