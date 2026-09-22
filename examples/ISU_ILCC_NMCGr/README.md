# ISU-ILCC NMC-Gr Data set
Paul Gasper, NLR, 2026

Download raw data from https://doi.org/10.25380/iastate.22582234. The original authors provide some example data processing code at https://doi.org/10.5281/zenodo.10648587, some of which is copied here. Uncompressed raw data files and subfolders should then all be placed in `data_raw` folder, for example, there should be a `data_raw/C_rates_and_DoD.csv` file and `data_raw/RPT_json` folder. Some previous work by NLR on this data set has been published at https://iopscience.iop.org/article/10.1149/1945-7111/adcb6f/meta with code and data reproduced at https://zenodo.org/records/14285817.

`battimal` data processing code for this data set is defined in `ISUILCC_NMCGr_Data.py`. Example Jupyter notebooks show the structure of the processed data on a subset of the cells. The script `process_data.py` reads in the raw data, exports matrices structured for training various model types, and saves the processed data as an .h5 data base.
