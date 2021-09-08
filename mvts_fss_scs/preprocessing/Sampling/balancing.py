
import os
import sys
sys.path.append("/home/ayeolekar1/mvts_fss_scs")
from collections import OrderedDict
import numpy as np
import pandas as pd
from CONSTANTS import PREPROCESSED_DATA_SAMPLES, SAMPLED_DATA_SAMPLES
from mvts_fss_scs.fss.utils import get_class_mapping
from mvts_fss_scs.preprocessing.Sampling.sampler import \
    Sampler as climatology_sampler
from tqdm import tqdm

#TODO: Add paths to refactor code


SAVE_PATH = SAMPLED_DATA_SAMPLES




class Sampler:

  def __init__(self, partitions,desired_ratios = None, label_col = "FLARE_CLASS"):

    self.partitions = partitions
    self.desired_ratios = desired_ratios
    self.partition_1, self.partition_2, self.partition_3, self.partition_4, self.partition_5 = self.partitions

    self.label_col = label_col

    self.mapping = {
      "B": 0,
      "C": 0,
      "F": 0,
      "M": 1,
      "X": 1
    }

  

  def get_stacked_training_keys(self):
    
    training_data = self.partition_2['np_data']
    training_labels = self.partition_2['subclass']
    return training_data, training_labels



  def get_id_label_mapping(self):

    id_label = pd.DataFrame(columns = ['ID',self.label_col])
    training_data, training_labels = self.get_stacked_training_keys()
    


    for file in tqdm(range(training_data.shape[0])):

      row = [file, training_labels[file]]
      id_label.loc[len(id_label),:] = row

    return id_label

  def get_sampled_data(self):



    dict_ratio = self.desired_ratios if self.desired_ratios else {'M': -1, 'X': -1, 'F': 0.013425132199138315, 'C': 0.001631244855612698, 'B': 0.0009214524238311712}



    id_label = self.get_id_label_mapping()

    sampler = climatology_sampler(id_label, label_col_name=self.label_col)
    id_label_sampled = sampler. sample(desired_ratios = dict_ratio)
    

    return id_label[id_label['ID'].isin(id_label_sampled['ID'])]
  
  def get_training_data(self):

    file_indices = self.get_sampled_data()["ID"].to_numpy()
    training_data,training_labels = self.get_stacked_training_keys()
    training_matrix = np.zeros((len(file_indices), 60,24))
    label_vector = np.empty((len(file_indices)),dtype='object')
    i=0
    
    for file in tqdm(file_indices):
      X = training_data[file,:,:]
      label_vector[i] = self.mapping[training_labels[file]]
      training_matrix[i] = X
      i+=1

    return training_matrix,label_vector

  def get_testing_data(self):
    
    testing_data = []
    testing_labels = []
    
    for partition in self.partitions[2:]:

      testing_data.append(partition['np_data'])

      testing_labels.append(list(map(get_class_mapping, partition['subclass'])))
      
    
    return testing_data, testing_labels




def get_combinations(partitions,k=0):
  # 1,2,3,4,5 k=0
  # 5,1,2,3,4 k=1
  # 4,5,1,2,3 K=2
  # 3,4,5,1,2 k=3
  # 2,3,4,5,1 k=4
  partitions = (partitions[-k:] + partitions[:-k])
  return partitions



def main():

  partition_1 = np.load(os.path.join(PREPROCESSED_DATA_SAMPLES, "partition1.npz"))
  partition_2 = np.load(os.path.join(PREPROCESSED_DATA_SAMPLES, "partition2.npz"))
  partition_3 = np.load(os.path.join(PREPROCESSED_DATA_SAMPLES, "partition3.npz"))
  partition_4 = np.load(os.path.join(PREPROCESSED_DATA_SAMPLES, "partition4.npz"))
  partition_5 = np.load(os.path.join(PREPROCESSED_DATA_SAMPLES, "partition5.npz"))
  partitions = [partition_1,partition_2,partition_3,partition_4,partition_5]
  s = Sampler(partitions)

  training_data, training_labels = s.get_training_data()


  np.savez_compressed(os.path.join(SAMPLED_DATA_SAMPLES,f"training.npz"), training_data=training_data, training_labels=training_labels)
  testing_data,testing_labels = s.get_testing_data()

  for k in range(len(testing_data)):

    print(testing_labels[k])

    np.savez_compressed(os.path.join(SAMPLED_DATA_SAMPLES,f"testing{str(k+3)}.npz"), testing_data=testing_data[k], testing_labels=testing_labels[k])
  # for k in range(5):
  #   partitions = [partition_1,partition_2,partition_3,partition_4,partition_5]
  #   partitions = get_combinations(partitions,k)
  #   partition_1,partition_2,partition_3,partition_4,partition_5 = partitions
  #   s = Sampler([partition_1,partition_2,partition_3,partition_4,partition_5])

    # training_data, training_labels = s.get_training_data()
    # testing_data, testing_labels = s.get_testing_data()
    # # print(training_labels)
    # # print(testing_labels)

    # np.savez_compressed(os.path.join(SAMPLED_DATA_SAMPLES,f"training{str(k+1)}.npz"), training_data=training_data, training_labels=training_labels)
    # np.savez_compressed(os.path.join(SAMPLED_DATA_SAMPLES,f"testing{str(k+1)}.npz"), testing_data=testing_data, testing_labels=testing_labels)


if __name__ == "__main__":
  main()



