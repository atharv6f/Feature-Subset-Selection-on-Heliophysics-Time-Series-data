
import logging

logging.basicConfig(filename='logger_file.log',level = logging.DEBUG, format = '%(levelname)s:%(asctime)s:%(filename)s')

import os
from collections import OrderedDict
from functools import lru_cache

import numba
import numpy as np
import pandas as pd
from dtaidistance import dtw
from matplotlib import pyplot as plt
from mvts_fss_scs.fss.base_fss import BaseFSS
from mvts_fss_scs.fss.pie.mutual_info import mi
from mvts_fss_scs.fss.utils import get_column_names
from numpy import linalg as LA
from scipy.sparse import csgraph
from scipy.stats import entropy
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding, spectral_embedding
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm


class PIE(BaseFSS):
  def __init__(self,data = None,**kwargs):

    if data:
      self.data = data
    else:
      super().__init__(data)

    self.n_neighbors = kwargs.pop('n_neighbors')
    self.mode = kwargs.pop('mode')
    self.mapping = {
      "B": 0,
      "C": 0,
      "F": 0,
      "M": 1,
      "X": 1
    }


  def get_keys(self):
    return self.data['np_data'], self.data['subclass']
 


    
  def eigenDecomposition(self, A, plot=True, topK=5):

    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)

    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values

    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters[0], eigenvalues, eigenvectors 

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

  
  def get_embed_vector(self, timeseries):


    """
    Step 1: Calculate DTW distance for a feature across all the timesteps
    Step 2: Generate K-Nearest Neighbour graph of the distance matrix
    Step 3: Convert the directed graph to acyclic graph by multiplying the edge weights
    Step 4: Fill all diagonal elements with "1"
    Step 5: Generate adjacency matrix for the distance graph
    Step 6: Generate embedding matrix using the adjacency matrix

    :param timeseries: Values of a feature across flare and non-flare samples
    :type timeseries: list
    :return: Embedded matrix of the feature
    :rtype: np.ndarray
    """

    ds = dtw.distance_matrix_fast(timeseries)
   
    ds = np.nan_to_num(ds)
    A = kneighbors_graph(ds, self.n_neighbors, mode=self.mode, include_self=True)
    A = 0.5 * (A + A.T)
    adj = A.toarray()
    
    # embed = ase.fit_transform(adj)
    # k,_,_ = self.eigenDecomposition(adj, plot=False)
    # cluster = SpectralClustering(n_clusters=k).fit(adj)
    # normzalized_adj = cluster.affinity_matrix_
    embedding = SpectralEmbedding(n_components=1, affinity = 'precomputed')
    # embed = spectral_embedding(normzalized_adj,n_components=1)
    embed = embedding.fit_transform(adj)
    return embed

  
  def relevance_score(self, embed, y):
    """
    Calculates the relevance score using the formula specified in the paper

    :param embed: Embedding matrix of a feature
    :param y: Label encoded ground truth labels
    :return: Score calculated based on the formula in the paper
    :rtype: np.float64
    """
    
    mututal_information = mi(y, embed)
    entropy_embed = entropy(embed)
    entropy_y = entropy(y)
    relevance_score = mututal_information / (np.sqrt(entropy_embed * entropy_y))
    return relevance_score

  @lru_cache(maxsize=8)
  def rank(self):
    """
    Returns a dataframe with all the sensors as index and their respective relevance score as column

    :return: Dataframe with Column name as index and relevance score as column
    :rtype: pandas.core.frame.DataFrame
    """

    column_mapping = self.__get_column_mapping()
    timeseries = []
    y = []
    scores = {}
    sampled_data,labels = self.get_keys()

    for col in tqdm(range(sampled_data.shape[2])):
      for file in tqdm(range(sampled_data.shape[0])):
        timeseries.append(sampled_data[file,:,col])
        
        y.append(self.mapping[labels[file]])


      embed = self.get_embed_vector(timeseries)
      embed_y = KMeans(n_clusters=5).fit_predict(embed)
      y = np.array(y).flatten() 

      scores[column_mapping[col]] = self.relevance_score(embed_y, y)
      timeseries = []
      y = []

    print("Embedding matrix generated")
    print("Scores Computed")  
    return pd.DataFrame(scores.items(), columns = ['Feature','Score']).sort_values(by='Score', ascending=False).reset_index(drop=True)
    

