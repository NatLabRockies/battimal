# MATR NMC-Gr Formation Data set
Paul Gasper, NLR, 2026


Download raw data from https://data.matr.io/8/. Uncompressed raw data files and subfolders should then all be placed in `data_raw` folder, for example, there should be several `.csv` files with file names like `Formation_2022-Parameter.csv` and folders `Aging` and `Formation`. The raw data from the original authors is in a `.json` format after processing via the `BEEP` python library; use the provided `BEEP DATA PROCESSING.ipynb` code with a BEEP-specific virtual environment (requirements detailed in `BEEP ENVIRONMENT REQUIREMENTS.txt`) to extract data from the BEEP format, which must be read using the BEEP virtual environment, into a more generally readable `.pkl` files.

`battimal` data processing code for this data set is defined in `MATR_NMCGr_Formation_Data.py`. Example Jupyter notebooks show the structure of the processed data on a subset of the cells. The script `process_data.py` reads in the raw data, exports matrices structured for training various model types, and saves the processed data as an .h5 data base.
