import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# RAW_DATA_PATH = "/home/sagar/Data/Data_Archives/SCS/raw"
# PROCESSED_DATA_PATH = "/home/sagar/Data/Data_Archives/SCS/processed"

RAW_DATA_PATH = "/home/spatel389/data/SCS_mini/raw/"
PROCESSED_DATA_PATH = "/home/spatel389/data/SCS_mini/processed/"


def get_files(path, extention):
    """Given directory path and extention - returns list of files within the directory including subdirectories"""
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                file_list.append(os.path.abspath(os.path.join(root, file)))
    return file_list


def read_npz(file):
    npzfile = np.load(file)
    return npzfile['np_data'], npzfile['target'], npzfile['subclass'], npzfile['file_path']


def make_global_npz(files):
    return(np.vstack([read_npz(file)[0] for file in files]))


def get_scaler(complete_data):
    reshaped_data = complete_data.reshape((-1,24))
    scaler = MinMaxScaler()
    scaler.fit(reshaped_data)
    return scaler, reshaped_data

def main():
    start_time = perf_counter()

    files = get_files(RAW_DATA_PATH, ".npz")

    complete_data = make_global_npz(files)
    scaler, reshaped_data = get_scaler(complete_data)

    # print("Range")
    # print(np.max(reshaped_data, axis=0) - np.min(reshaped_data, axis=0))


    for file in files:
        print(file)
        np_data, target, subclass, file_path = read_npz(file)
        reshaped_data = np_data.reshape(-1,24)
        scaled_data = scaler.transform(reshaped_data)
        recon_data = scaled_data.reshape(np_data.shape[0],np_data.shape[1],np_data.shape[2])
        export_file = PROCESSED_DATA_PATH + os.sep + file.split(os.sep)[-1]
        np.savez_compressed(export_file, np_data=recon_data, target=target, subclass=subclass, file_path=file_path)


    end_time = perf_counter()
    time_elapsed = end_time - start_time
    print(f'Total time {time_elapsed} seconds')


if __name__ == '__main__':
    main()
