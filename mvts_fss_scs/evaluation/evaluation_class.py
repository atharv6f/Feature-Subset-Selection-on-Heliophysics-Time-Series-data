import multiprocessing
import os
import timeit
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import sys 
# sys.path.append("/home/ayeolekar1/mvts_fss_scs")

sys.path.append("C:\\Users\\ayeolekar1\\Desktop\\mvts_fss_scs")

import numpy as np
import pandas as pd
from CONSTANTS import RESULTS, SAMPLED_DATA_SAMPLES
from mvts_fss_scs.evaluation.metrics import *
from mvts_fss_scs.evaluation.metrics import calc_hss2, calc_tss
from mvts_fss_scs.fss.utils import get_column_names, save_table
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from tslearn.svm import TimeSeriesSVC


class Evaluation:


  def __init__(self, feature_set, sampled_file_path =None, kernel='linear'):

    if sampled_file_path:
      self.sampled_file_path = sampled_file_path

    else:
      self.sampled_file_path = SAMPLED_DATA_SAMPLES

    self.kernel = kernel
    self.feature_set = feature_set
    self.univariate_result = pd.DataFrame(columns = ['Feature', 'Mean_TSS', 'Mean_HSS', 'Std_TSS', 'Std_HSS'])
    self.multivariate_result = pd.DataFrame(columns = ['Num_Features', 'Mean_TSS', 'Mean_HSS', 'Std_TSS', 'Std_HSS'] )
    self.clf = TimeSeriesSVC(kernel = self.kernel, n_jobs = -1)
    self.training = np.load(os.path.join(self.sampled_file_path, "training.npz"), allow_pickle=True)
    self.X_train, self.y_train = self.training['training_data'], self.training['training_labels']
    self.X_train = np.nan_to_num(self.X_train)
    self.tss_scores_uni =  defaultdict(list, {k:[] for k in get_column_names()})
    self.hss_scores_uni =  defaultdict(list, {k:[] for k in get_column_names()})

  # def __get_train_test(self, training, testing):
  #   return training['training_data'], training['training_labels'], testing['testing_data'], testing['testing_labels']



  def __get_column_mapping(self):

    """
    Private method that maps the columns in the file to an integer index
    :returns Mapping from column to index
    :rtype dict
    """
    cols = get_column_names()
    column_mapping = {}
    for i in range(len(cols)):
      column_mapping[i] = cols[i]

    return column_mapping

  def __reverse_column_mapping(self):
    column_mapping = self.__get_column_mapping()
    new_mapping = {}
    for key, value in column_mapping.items():
      if value in new_mapping:
        new_mapping[value].append(key)
      else:
        new_mapping[value] = key

    return new_mapping


  def compute_tss_hss(self,clf, X_train, y_train, X_test, y_test, feature, part_id):
    clf.fit(X_train, y_train.astype("int"))
    y_pred = clf.predict(X_test)
    tss = calc_tss(y_test, y_pred)
    hss = calc_hss2(y_test, y_pred)
    return {'feature':feature, "part_id":part_id, 'tss':tss, 'hss':hss}




  def univariate(self):
    entries = []
    with ProcessPoolExecutor(max_workers=24) as executor:
      futures = []
      reverse_column_mapping = self.__reverse_column_mapping()

      for i in tqdm(range(3,6)):
        testing = np.load(os.path.join(self.sampled_file_path, f"testing{i}.npz"), allow_pickle=True)
        X_test, y_test = testing['testing_data'],testing['testing_labels']

      for feature in self.feature_set['Feature']:

        j = reverse_column_mapping[feature]
        futures.append(executor.submit(
                self.compute_tss_hss, clf=self.clf, X_train=self.X_train[:,:,j], y_train=self.y_train, X_test=X_test[:,:,j], y_test=y_test, feature=feature, part_id=i))
    for future in as_completed(futures):
        entries.append(future.result())

    # for feature in self.feature_set['Feature']:
    #   row = [feature, np.mean(self.tss_scores_uni[feature]), np.mean(self.hss_scores_uni[feature]), np.std(self.tss_scores_uni[feature]), np.std(self.hss_scores_uni[feature])]
    #   self.univariate_result.loc[len(self.univariate_result),:] = row

    return pd.DataFrame(entries)

  def multivariate(self):

    reverse_column_mapping = self.__reverse_column_mapping()

    tss_scores =  defaultdict(list, {k:[] for k in range(24,0,-1)})
    hss_scores =  defaultdict(list, {k:[] for k in range(24,0,-1)})

    for i in tqdm(range(3,6)):
      testing = np.load(os.path.join(self.sampled_file_path, f"testing{i}.npz"), allow_pickle=True)
      X_test, y_test = testing['testing_data'],testing['testing_labels']
      num_features = 24
      feature_index = []
      while num_features > 0:
        features = list(self.feature_set['Feature'][:num_features])

        for feature in features:
          feature_index.append(reverse_column_mapping[feature])


        self.clf.fit(self.X_train[:,:,feature_index], self.y_train.astype("int"))
        y_pred = self.clf.predict(X_test[:,:,feature_index])

        # print(i, num_features, calc_tss(y_test, y_pred))
        tss_scores[num_features].append(calc_tss(y_test, y_pred))
        hss_scores[num_features].append(calc_hss2(y_test, y_pred))


        num_features -= 1

    for feature in range(24,0,-1):
      row = [feature, np.mean(tss_scores[feature]), np.mean(hss_scores[feature]),np.std(tss_scores[feature]), np.std(hss_scores[feature])]
      self.multivariate_result.loc[len(self.multivariate_result),:] = row

    return self.multivariate_result
        # print(self.X_train[:,:,self.feature_set[:num_features].index].shape)
        # num_features -= 1
        # self.clf.fit(self.X_train[:,:,self.feature_set['Feature'].index[:num_features]])
        # y_pred = self.clf.predict(X_test[:,:self.feature_set['Feature'][:num_features]])
        # print(num_features)
        # tss_scores[number].append(calc_tss(y_test, y_pred))
        # hss_scores[feature].append(calc_hss2(y_test, y_pred))






def main():





  #pie_rank = pd.read_csv(os.path.join(RESULTS,'pie_rank.csv')).reset_index()
  csfs_rank = pd.read_csv(os.path.join(RESULTS,'csfs_rank.csv')).reset_index()
  #fcbf_rank = pd.read_csv(os.path.join(RESULTS,'fcbf_rank.csv')).reset_index()
  eval = Evaluation(csfs_rank, kernel='linear')

  uni = eval.univariate()

  print(uni)
  # multi = eval.multivariate()

  # save_table(RESULTS, uni, "univariate_csfs")
  # save_table(RESULTS, multi, "multivariate_csfs")

  # Univariate
  # GAK = 1 min 9sec
  # RBF = 6 seconds


if __name__ == '__main__':
    main()


