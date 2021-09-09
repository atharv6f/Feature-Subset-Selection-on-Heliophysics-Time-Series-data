import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
import os
import numpy as np
import pandas as pd


SWAN_DATA_PATH = "/home/spatel389/data/SWAN-SF/"
RAW_DATA_PATH = "/home/spatel389/data/SCS_mini/raw/"

# SWAN_DATA_PATH = "D:/Data Archives/SWAN-SF/"
# RAW_DATA_PATH = "D:/Data Archives/scs_raw/"


# To avoid location based indexing and ensuring consistency in data extraction
first24 = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
           'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM',
           'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
           'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE']

# To ensure replicability
random_states = [266, 767, 613,  15, 880, 232, 688, 381, 837, 555]


def get_files(path, extention):
    """Given directory path and extention - returns list of files within the directory including subdirectories"""
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                file_list.append(os.path.abspath(os.path.join(root, file)))
    return file_list


def get_flat_data(file, columns):
    """Given a file path - returns the dictionary of selected columns along with target and file path"""
    df = pd.read_csv(file, sep="\t")
    # Linear interpolation
    df.interpolate(method='linear', limit_direction="both", inplace=True)
    
    flat_data = {}
    for col in columns:
        flat_data.update({col+"_"+str(idx): value for idx,
                     value in enumerate(df[col])})
    flat_data['file'] = file
    flat_data['subclass'] = file.split(os.sep)[-1][0]
    flat_data['target'] = file.split(os.sep)[-2]
    return flat_data


def get_np_data(file, columns):
    """Given a file path - returns the dictionary of selected columns along with target and file path"""
    df = pd.read_csv(file, sep="\t")
    # Linear interpolation
    df.interpolate(method='linear', limit_direction="both", inplace=True)
    df = df[columns]

    if df.notna().values.all():
        data = {
            "numpy_data" : df.to_numpy(),
            "subclass" : file.split(os.sep)[-1][0],
            "target" : file.split(os.sep)[-2],
            "file" : file
        }
        return data


def parallel_parse_df(file_list, fun):
    entries = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for file in file_list:
            futures.append(executor.submit(
                fun, file=file, columns=first24))
        for future in as_completed(futures):
            entries.append(future.result())
    df = pd.DataFrame(entries)
    return df


def parallel_parse_np(file_list, fun):
    entries = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for file in file_list:
            futures.append(executor.submit(
                fun, file=file, columns=first24))
        for future in as_completed(futures):
            entries.append(future.result())
    np_data = np.stack([entry['numpy_data'] for entry in entries if entry is not None])
    target = [entry['target'] for entry in entries if entry is not None]
    subclass = [entry['subclass'] for entry in entries if entry is not None]
    file = [entry['file'] for entry in entries if entry is not None]
    return np_data, target, subclass, file


def main():
    start_time = perf_counter()

    for i in range(1,6):
        print(f"Parsing partition {i}")
        files = get_files(SWAN_DATA_PATH+"partition"+str(i), ".csv")
        files = files[::100]
        flat_df = parallel_parse_df(files, get_flat_data)
        flat_df.to_csv(RAW_DATA_PATH + "partition" + str(i) + ".csv", index=False)

        np_data, target, subclass, file_path = parallel_parse_np(files, get_np_data)
        np.savez_compressed(RAW_DATA_PATH + "partition" + str(i) + ".npz", np_data=np_data, target=target, subclass=subclass, file_path=file_path)


    end_time = perf_counter()
    time_elapsed = end_time - start_time
    print(f'Total time {time_elapsed} seconds')


if __name__ == '__main__':
    main()
