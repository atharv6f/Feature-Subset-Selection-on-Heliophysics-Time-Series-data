import numpy as np
import pandas as pd
from fss.base_fss import BaseFSS
from fss.fcbf.fcbf_helper import *
from fss.pie.mutual_info import mi
from fss.utils import *
from sklearn import preprocessing


class FCBF(BaseFSS):
  """
  """

  def __init__(self,data = None,**kwargs):
  
    if data:
      self.data = data
    else:
      super().__init__(data)
    
    self.req_data = self.data['np_data']
    self.Y = pd.Series(np.repeat(self.data['subclass'], 60), name= 'subclass')

    """
    1. This is where use initialize the keyword arguments you need for your algorithm.
    Refer pie.py or csfs.py to get a rough idea of the implementation.

    2. You need to implement a method called rank which returns your scores(dataframe) in desceneding order
      in the following format.

      Features | Score

    3. Create an instance of the above class in the main __init__file
    """

  def relevant_features(self):
    """
    A method to find relevant features
    """

    rel_features = {}

    for feature in range(24):
      X = pd.Series(self.req_data[:,:,feature].ravel(), name= get_column_names()[feature])
      SUxc = symmetrical_uncertainty(X, self.Y)
      rel_features[get_column_names()[feature]] = SUxc
      sorted_rel_features = {k: v for k, v in sorted(rel_features.items(), key=lambda item: item[1], reverse=True)}

    print(sorted_rel_features)
    return sorted_rel_features

  def relevant_features_list(self) -> list:
        """
        This method returns list of relevant features. 
        Symmetrical entropy values are removed and only features are kept.

        :return list: Returns list of the features. [feature1, feature2, ...]
        """
        return  list(self.relevant_features().keys())
    

  def rank(self):
    """
    Dataframe Output
    """
    df = pd.DataFrame(self.relevant_features().items(), columns= ['Feature', 'Score'])
    return df
  
