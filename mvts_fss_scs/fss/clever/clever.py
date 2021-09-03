from mvts_fss_scs.fss.base_fss import BaseFSS
from mvts_fss_scs.fss import utils
import pandas as pd
import numpy as np
from scipy import linalg
import os
from numpy import linalg as LA


def get_file_name(file_dir_lst):
    """
  :param file_name: File name
  :return: file name with file type cropped
  """
    files_list = []
    for dir in file_dir_lst:
        contents = dir.split('/')
        files_list.append(contents[7][:-4])
    return files_list


def SVD(ndf):
    """
    SVD - Singular value Decomposition
    :param ndf: Dataframe
    :return: Singular Vectors, Variance
    """
    U, s, Vh = linalg.svd(ndf)
    return U, s, Vh


def percent_variance(var):
    """
  variances percentages
  :param var: variances (singular or eigen values)
  :return: list of variance percentages
  """
    percentv = []
    total = np.sum(var)
    for p in var:
        val = 100 * (p / total)
        percentv.append(val)
    return percentv


def num_var_lessthan_threshold(percentv, threshold):
    """
    gets the number of variables whose sum is less than the threshold value chosen
    :param percentv: List of variable percentages
    :param threshold: threshold value ( 70% - 90% )
    :return: count - sum of p variables percentages and len(p) - no. of variables
    """
    count = 0
    p = []
    for indx, x in enumerate(percentv):
        if count <= threshold:
            count = count + x
            p.append(indx)
    return count, len(p)


def loadings_files(loadings_path, load, file_name):
    """
  Saving loadings to a csv file
  :param loadings_path: path to save the file
  :param load: loadings (singular vectors or eigen vectors)
  :param file_name: File name
  :return: returns the dataframe
  """
    load.to_csv(loadings_path + file_name, sep="\t", index=False)


def loadings_data_load(data_path: str):
    """
    To read the loading files stored
    :param data_path: path to the directory
    :return: returns a dataframe
    """
    sample_file = pd.read_csv(data_path, sep='\t')
    dataframe = pd.DataFrame(sample_file)
    return dataframe


def List_Of_Files(dir_name):
    """
    List of files in the given directory
    :param dir_name: path of the directory
    :return: A list of files in the directory
    """
    listOfFile = [x[2] for x in os.walk(dir_name)]
    return listOfFile[0]


def Sort(sub_li):
    """
    Sort the list of lists based on the second element
    :param sub_li: list of lists
    :return: returns the sorted list of lists
    """
    sub_li.sort(key=lambda x: x[1], reverse=True)
    return sub_li


def dict_features(all_features):
    feature_dict = {}
    for ind, feature in enumerate(all_features):
        feature_dict[ind] = feature
    return feature_dict


class CLEVER(BaseFSS):

    def __init__(self, data=None, **kwargs):

        if data:
            self.data = data
        else:
            super().__init__(data)
        self.n_features = len(utils.get_column_names())
        self.features = utils.get_column_names()

        """
    1. This is where use initialize the keyword arguments you need for your algorithm.
    Refer pie.py or csfs.py to get a rough idea of the implementation.

    2. You need to implement a method called rank which returns your scores (dataframe) in desceneding order
      in the following format.

      Features | Score

    3. Create an instance of the above class in the main __init__file
    """

    def Compute_DCPCS(self):
        """
    Computes principal components (PCs) and common principal components (DCPCs)
    :param path: path to the main directory
    :return: returns DCPCs
    """

        count = 0
        P_values_each_file = pd.DataFrame(columns=["percentvar", 'no_of_variables'])
        loadings_path = os.path.join(os.getcwd(), "fss\\clever\\Loadings\\")
        #loadings_path = r"C:\\Users\\Krishna Rukmini\\PycharmProjects\\SummerCodeSprint\\mvts_fss_scs\\mvts_fss_scs\\fss\\clever\\Loadings\\"
        #file_name = get_file_name(CONSTANTS.part1['file_path'][0:2])
        for ft in self.data['np_data']:
            data = pd.DataFrame(ft)
            corrMatrix = data.corr()
            # if corrMatrix.isnull().values.any():
            # print(file_name[count])
            # else:
            corrMatrix = corrMatrix.fillna(0)
            U, s, Vh = SVD(corrMatrix)
            Loadings = U
            Variance = s
            prcnt_var = percent_variance(Variance)
            to_append = list(num_var_lessthan_threshold(prcnt_var, 90))
            a_series = pd.Series(to_append, index=P_values_each_file.columns)
            P_values_each_file = P_values_each_file.append(a_series, ignore_index=True)
            Loadings_df = pd.DataFrame(Loadings)
            Loadings_p = loadings_path + str(count) + ".csv"
            Loadings_df.to_csv(Loadings_p, sep="\t", index=False)
            count += 1
        data_col = dict_features(self.features)
        list_of_files = List_Of_Files(loadings_path)
        p = int(max(P_values_each_file['no_of_variables']))
        H_matrix = pd.DataFrame(index=range(24), columns=range(24))
        H_matrix[:] = 0
        for file in list_of_files:
            data = loadings_data_load(loadings_path + file)
            Loading = data[:p][:]
            Loading_T = Loading.T
            H_matrix = np.matrix(np.add(H_matrix, np.matmul(np.asarray(Loading_T), np.asarray(Loading))), dtype=float)
        new_u, new_s, new_Vh = SVD(H_matrix)
        DCPC = pd.DataFrame(new_u[:p][:])
        DCPC.to_csv(os.path.join(os.getcwd(), "fss/clever/DCPC.csv"), sep='\t')
        return DCPC, data_col

    def rank(self):
        pair = []
        selected_var = []
        DCPC, col = self.Compute_DCPCS()

        for ind in DCPC.columns:
            val = LA.norm(DCPC[ind])
            pair.append([ind, val])
        Sort(pair)
        print(pair)
        selected_var_dist = pair[:self.n_features]
        for ind, close in enumerate(selected_var_dist):
            selected_var_dist[ind][0] = col[close[0]]
            selected_var.append(selected_var_dist[ind][0])
        ranks_dict = {"Features": selected_var, "Score": list(range(1, self.n_features+1))}
        return pd.DataFrame(ranks_dict).sort_values(by='Score', ascending=True).reset_index(drop=True)
