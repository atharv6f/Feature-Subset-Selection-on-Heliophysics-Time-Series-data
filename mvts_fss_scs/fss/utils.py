
import os

import numpy as np
import pandas as pd


def save_table(path, table, table_name, index=False, header=True):
    """
    Stores table to the specified path
    :param index: Write row names if true, false otherwise
    :param header: Write column name if true, false otherwise
    :param table: Dataframe to be saved
    :param table_name: "Table will be saved with this name"
    :param path: Location where the dataframe is to be stored
    :type table: pandas.core.frame.DataFrame
    :type table_name: str
    :type index: bool
    :type header: bool
    :returns: None
    """
    table.to_csv(os.path.join(path, table_name + ".csv"), index=index, header=header)


def update_results(df, path):
    with open(path, 'a') as f:
        df.to_csv(f, header=f.tell()==0, index=False)


def prepare_path(*args):
    return "_".join(args)

def get_column_names():
    return ["TOTUSJH","TOTBSQ","TOTPOT","TOTUSJZ","ABSNJZH","SAVNCPP","USFLUX","TOTFZ","MEANPOT","EPSZ","MEANSHR","SHRGT45","MEANGAM","MEANGBT","MEANGBZ","MEANGBH","MEANJZH","TOTFY","MEANJZD","MEANALP","TOTFX","EPSY","EPSX","R_VALUE"]




def get_class_mapping(labels):
    
    tranformed_labels = []
    for i in range(len(labels)):
        tranformed_labels.append(1 if labels[i] in ["M","X"] else 0)
    
    return tranformed_labels[0]
