import os
from abc import ABC, abstractmethod

import numpy as np
#from CONSTANTS import DATA_PATH
#DATA_PATH = "/home/ayeolekar1/mvts_fss_scs/mvts_fss_scs/preprocessed_data_samples"
# DATA_PATH = r"C:\Users\Krishna Rukmini\PycharmProjects\SummerCodeSprint\mvts_fss_scs\mvts_fss_scs\preprocessed_data_samples"
# DATA_PATH = r"D:\Projects\mvts_fss_scs\mvts_fss_scs\preprocessed_data_samples"
class BaseFSS(ABC):

  def __init__(self, *args):
    
   
    self.data = np.load(os.path.join(DATA_PATH,"partition1.npz"))
 
  
  @abstractmethod
  def rank(self):
    pass
