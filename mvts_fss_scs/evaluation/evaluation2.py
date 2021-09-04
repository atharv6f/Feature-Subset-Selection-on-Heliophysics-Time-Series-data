import multiprocessing
import os
import timeit
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import inspect
from time import perf_counter

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, maindir) 
import numpy as np
import pandas as pd
from mvts_fss_scs.evaluation.metrics import calc_hss2, calc_tss
from mvts_fss_scs.fss.utils import get_column_names, save_table, update_results, prepare_path
from tqdm import tqdm
from tslearn.svm import TimeSeriesSVC

def get_files(path, extention):
    """Given directory path and extention - returns list of files within the directory including subdirectories"""
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extention):
                file_list.append(os.path.abspath(os.path.join(root, file)))
    return file_list

MAX_WORKERS = 12
RANKS_PATH = "Results/ranks"
RESULTS_PATH = "Results/evaluations_test"
SAMPLED_DATA_PATH = "mvts_fss_scs/sampled_data"
TRAINING = "training2.npz"
IDX = str(TRAINING.split('.')[0][-1])
RANKING_FILES = [x.split('/')[-1] for x in get_files(RANKS_PATH,".csv")]

def __get_column_mapping():
  
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


def __reverse_column_mapping():
  column_mapping = __get_column_mapping()
  new_mapping = {}
  for key, value in column_mapping.items():
    if value in new_mapping:
      new_mapping[value].append(key)
    else:
      new_mapping[value] = key

  return new_mapping


def compute_tss_hss(clf, X_train, y_train, X_test, y_test, feature, part_id):
  clf.fit(X_train, y_train.astype("int"))
  y_pred = clf.predict(X_test)
  tss = calc_tss(y_test, y_pred)
  hss = calc_hss2(y_test, y_pred)
  # temp_df = pd.DataFrame({'feature':[feature], "part_id":[part_id], 'tss':[tss], 'hss':[hss]})
  # update_results(temp_df, os.path.join(RESULTS_PATH,"logs",prepare_path(EVALUATION_TYPE,RANKING_ALGO,KERNEL)+".csv")) 
  return {'feature':feature, "part_id":part_id, 'tss':tss, 'hss':hss}



def calculate_stats(sample, univariate = True):
  
  stats = sample.groupby(['feature'], as_index=False).agg({'tss':['mean', 'std'], 'hss': ['mean','std']})
  stats.columns = stats.columns.droplevel(0)
  if univariate:
    stats.columns = ["FEATURE","MEAN_TSS", "STD_TSS", "MEAN_HSS", "STD_HSS"]

  else:
    stats.columns = ["NUM_FEATURES","MEAN_TSS", "STD_TSS", "MEAN_HSS", "STD_HSS"]
    stats = stats.sort_values(by = "NUM_FEATURES", ascending=False)

  return stats.reset_index(drop=True)

def univariate(sampled_file_path, feature_set,clf,X_train,y_train):
  entries = []
  testing_sets = []
  for i in range(3,6):
    testing_sets.append(np.load(os.path.join(sampled_file_path, f"testing{i}.npz"), allow_pickle=True))

  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    reverse_column_mapping = __reverse_column_mapping()
    with tqdm(total=72) as pbar:
      for idx,testing in enumerate(testing_sets):
        X_test, y_test = testing['testing_data'],testing['testing_labels']
        futures = []

        for feature in feature_set['Feature']:

          j = reverse_column_mapping[feature]
          futures.append(executor.submit(
                  compute_tss_hss, clf=clf, X_train=X_train[:,:,j], y_train=y_train, X_test=X_test[:,:,j], y_test=y_test, feature=feature, part_id=idx+2))
        for future in as_completed(futures):
          entries.append(future.result())
          pbar.update(1)

  df = calculate_stats(pd.DataFrame(entries),univariate=True)
        
  return df


def multivariate(sampled_file_path, feature_set,clf,X_train,y_train):
  entries = []
  testing_sets = []
  for i in range(3,6):
    testing_sets.append(np.load(os.path.join(sampled_file_path, f"testing{i}.npz"), allow_pickle=True))

  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    reverse_column_mapping = __reverse_column_mapping()
    with tqdm(total=72) as pbar:
      for testing in testing_sets:
        futures = []
        X_test, y_test = testing['testing_data'],testing['testing_labels']
        num_features = 24
        while num_features > 0:
          feature_index = []
          features = list(feature_set['Feature'][:num_features]) 
          
          for feature in features:
            feature_index.append(reverse_column_mapping[feature])

          futures.append(executor.submit(
                  compute_tss_hss, clf=clf, X_train=X_train[:,:,feature_index], y_train=y_train, X_test=X_test[:,:,feature_index], y_test=y_test[:], feature=num_features, part_id=i))
          num_features-=1

        for future in as_completed(futures):
          entries.append(future.result())
          pbar.update(1)

  df = calculate_stats(pd.DataFrame(entries), univariate=False)
  return df

def main():
  start_time = perf_counter()

  training = np.load(os.path.join(SAMPLED_DATA_PATH, TRAINING), allow_pickle=True)
  X_train, y_train = training['training_data'], training['training_labels']

  for KERNEL in ['linear','rbf']:
    for RANKING_FILE in RANKING_FILES:

      RANKING_ALGO = "_".join(RANKING_FILE.split("_")[:-1])

      feature_set = pd.read_csv(os.path.join(RANKS_PATH,RANKING_FILE)).reset_index()
      clf = TimeSeriesSVC(kernel = KERNEL)
      
      print(f"{TRAINING=}, {KERNEL=}, {RANKING_FILE=}, {RANKING_ALGO=}, {IDX=}")
      print(f"Univariate")
      univariate_result = univariate(sampled_file_path = SAMPLED_DATA_PATH, feature_set=feature_set, 
                      clf=clf, X_train=X_train, y_train=y_train)
      save_table(RESULTS_PATH, univariate_result, prepare_path("univariate",KERNEL,RANKING_ALGO,IDX))

      print(f"Multivariate")
      multivariate_result = multivariate(sampled_file_path = SAMPLED_DATA_PATH, feature_set=feature_set, 
                      clf=clf, X_train=X_train, y_train=y_train)
      save_table(RESULTS_PATH, multivariate_result, prepare_path("multivariate",KERNEL,RANKING_ALGO,IDX))


  time_elapsed = perf_counter() - start_time
  print(f'Total time {time_elapsed} seconds')
    
  
if __name__ == '__main__':
  main()


