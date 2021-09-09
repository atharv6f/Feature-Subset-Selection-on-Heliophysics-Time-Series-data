import numpy as np
import pandas as pd
from fss import utils
from fss.base_fss import BaseFSS
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


class CORONA(BaseFSS):
  """
  This class is implementation of CORONA algorithm:
    1. It creates the correlation coefficient matrix for MTS item.
    2. Vectorizes the above obtained matrix.
    3. Assign weights to matrix using SVM.
    4. Ranks the variables and eliminates based on recursion.
    5. steps 3 & 4 are performed together by training SVM and RFE model.
  """
  def __init__(self,data = None,**kwargs):
  
    if data:
      self.data = data
    else:
      super().__init__(data)
    self.n_features = len(utils.get_column_names())

  def feature_matrix(self):
    """
      This method creates a 2d array of size N mts items x length of mts vector.
      :return: feature matrix
    """
    feature_matrix = []
    for mts_data in self.data['np_data']:
      vector = np.array(self.vectorize(mts_data)).flatten()
      feature_matrix.append(vector)
    return feature_matrix

  def vectorize(self, mts_data):
    """
      This method converts the data to correlation matrix
      and vectorizes the upper triangle of correlation coefficient matrix.
      The 60*24 data is converted to
      264( 24*24(correlation matrix) % 2(only right triangle of symmetric matrix) - 1(exclude diagonal)) vector.
      :return: vectorized data
    """
    correlation_matrix = np.dot(mts_data, mts_data.T)
    upper_index = np.triu_indices(self.n_features, 1)
    return correlation_matrix[upper_index]

  def de_vectorize_ranks(self, ranks):
    """
    vector ranks are converted into symmetric matrix and
    obtain the ranks of each feature by column-wise summation.
    The 264 vector is obtained which is converted into 24*24 symmetric matrix
    by adding 264 values of vector in right and left triangle of matrix and diagonal
    is filled with 1.00.
    :param ranks: ranks obtained svm.
    :return: ranks of features.
    """
    symmetric_matrix = np.zeros((self.n_features, self.n_features))
    # right traingle
    symmetric_matrix[np.triu_indices(self.n_features, 1)] = ranks
    # left triangle
    symmetric_matrix[np.tril_indices(self.n_features, -1)] = ranks
    # diagonal
    row, col = np.diag_indices(self.n_features)
    symmetric_matrix[row, col] = np.array([1.00])
    return np.sum(symmetric_matrix, axis=0)

  def rank(self):
    """
    Gets the features, initializes the SVC linear model and RFE. Trains the features
    their respective classes on SVC and RFE and returns ranking_ object which is de-vectorized
    to get the ranks of each feature.
    :return: dataframe of features and scores.
    """
    features = self.feature_matrix()
    print("Training svc model")
    estimator = SVC(kernel='linear')
    rfe = RFE(estimator, self.n_features, step=1)
    ranks = rfe.fit(features, self.data['target'])
    final_ranks = self.de_vectorize_ranks(ranks.ranking_)
    columns = utils.get_column_names()
    ranks_dict = {"Features": columns, "Score": final_ranks}
    return pd.DataFrame(ranks_dict).sort_values(by='Score', ascending=False).reset_index(drop=True)
