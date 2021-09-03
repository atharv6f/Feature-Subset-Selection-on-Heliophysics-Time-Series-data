import os
from functools import lru_cache

import numba
import numpy as np
import pandas as pd
from fss.base_fss import BaseFSS
from fss.pie.mutual_info import mi
from fss.utils import get_column_names

from tqdm import tqdm


class CSFS(BaseFSS):

	def __init__(self,data=None,**kwargs):
  		
		if data:
			self.data = data
		else:
			super().__init__(data)

	def get_keys(self):
    
		return self.data['np_data'],self.data['target'], self.data['subclass']

	# def get_cols(self):
  		
	# 	cols = list(pd.read_csv(os.path.join(self.flare_path,  "M1.0@265_Primary_ar115_s2010-08-06T06_36_00_e2010-08-06T18_24_00.csv"),sep="\t").columns[1:25])
	# 	return np.array(cols)

	def get_upper_triangle(self, matrix):
  		
      #get the upper triangular part of this matrix
		v = matrix[np.triu_indices(matrix.shape[0], k = 0)]
		size_X = 24
		X = np.zeros((size_X,size_X))
		X[np.triu_indices(X.shape[0], k = 0)] = v
		X = X + X.T - np.diag(np.diag(X))
		return X

	@lru_cache(16)
	def get_flare_matrix(self):
		
		# if flare_class == "FL":
  	# 		print("Computing Pairwise Information for all flare samples")

		# else:
  	# 		print("Computing Pairwise Information for all non-flare samples")

		sampled_data, flare_class, labels = self.get_keys()
		
		flare_index = np.array(np.where(flare_class == "FL")).flatten()
		non_flare_index = np.array(np.where(flare_class == "NF")).flatten()
		

		flare_mat = np.zeros((len(flare_index), 24, 24))
		non_flare_mat = np.zeros((len(non_flare_index), 24, 24))

		i=0
		j=0

		for file in tqdm(flare_index):
  			
			X = sampled_data[file,:,:]
			X = np.nan_to_num(X)
			n = X.shape[1]
			matMI = np.zeros((24,24))
			for ix in np.arange(n):
				for jx in np.arange(ix+1,n):
					matMI[ix,jx] = mi(X[:,ix], X[:,jx])
				
			flare_mat[i] = self.get_upper_triangle(matMI)
			i+=1	


		for file in tqdm(non_flare_index):
  			
			X = sampled_data[file,:,:]
			X = np.nan_to_num(X)
			n = X.shape[1]
			matMI = np.zeros((24,24))
			for ix in np.arange(n):
				for jx in np.arange(ix+1,n):
					matMI[ix,jx] = mi(X[:,ix], X[:,jx])
				
			non_flare_mat[j] = self.get_upper_triangle(matMI)
			j+=1	
		

		return flare_mat, non_flare_mat

	def get_means_flare(self,i_flare, n_flare):
  		
		mean_flare = np.ones((n_flare, 24))
		for flare_index in range(n_flare):
				for i in range(24):
						mean_flare[flare_index][i] = np.mean(i_flare[flare_index, :, i])

		return mean_flare

	def get_means_non_flare(self,i_non_flare, n_non_flare):
		mean_non_flare = np.ones((n_non_flare, 24))
		for non_flare_index in range(n_non_flare):
				for i in range(24):
						mean_non_flare[non_flare_index][i] = np.mean(i_non_flare[non_flare_index, :, i])

		return mean_non_flare

	def get_means_total(self,mean_flare, mean_non_flare):
		mean_total = np.ones((24,1))
		for col in range(24):
				mean_total[col] = (mean_flare[:,col].sum()+ mean_non_flare[:,col].sum())/2
		
		return mean_total


	def s_b_matrix(self,i_flare, i_non_flare):
		s_b = np.ones((24,))
		
		n_flare= i_flare.shape[0]
		n_non_flare = i_non_flare.shape[0]
		
		mean_flare = self.get_means_flare(i_flare,n_flare)
		mean_non_flare = self.get_means_non_flare(i_non_flare, n_non_flare)
		mean_total = self.get_means_total(mean_flare, mean_non_flare)
		
		
		for i in tqdm(range(24)):
			print(f"Working on sensor {i+1}")
			u_flare = mean_flare[:,i]
			u_non_flare = mean_non_flare[:,i]
			a_flare = ((u_flare - mean_total[i]).reshape(1,-1))
			a_non_flare = ((u_non_flare - mean_total[i]).reshape(1,-1))

			s_b[i] = n_flare*(np.dot(a_flare,a_flare.T).flatten()[0]) + n_non_flare* (np.dot(a_non_flare,a_non_flare.T).flatten()[0])
			
		return s_b

	def s_w_matrix(self,i_flare, i_non_flare):

		s_w = np.ones((24,))
		n_flare= i_flare.shape[0]
		n_non_flare = i_non_flare.shape[0]

		mean_flare = self.get_means_flare(i_flare,n_flare)
		mean_non_flare = self.get_means_non_flare(i_non_flare, n_non_flare)

		diff_non_flare = 0
		diff_flare = 0
		for col in range(24):
			
			print(f"Working on sensor {col+1}")
			for file in range(i_flare.shape[0]):
					diff_non_flare += (mean_non_flare[file,col] - i_non_flare[file,:,col])
					diff_flare += (mean_flare[file,col] - i_flare[file,:,col])
			
			s_w_non_flare = np.dot(diff_non_flare, diff_non_flare.T)
			s_w_flare = np.dot(diff_flare, diff_flare.T)
			s_w[col] = s_w_non_flare + s_w_flare
			diff_non_flare = 0
			diff_flare = 0
			
		return s_w

	def rank(self):
		flare_mat,non_flare_mat = self.get_flare_matrix() #1254,24,24
		print(flare_mat.shape)
		print(non_flare_mat.shape)
		print("Generating Between Class Scatter Matrix")
		s_b = self.s_b_matrix(flare_mat,non_flare_mat)
		print("Between Class Scatter Matrix Generated")
		print("Generating Within Class Scatter Matrix")
		s_w = self.s_w_matrix(flare_mat,non_flare_mat)
		print("Within Class Scatter Matrix Generated")
		cols = np.array(get_column_names())
		matrix_df = pd.DataFrame(columns = ['Feature','S_B', 'S_W'], data = np.hstack((cols.reshape(-1,1),s_b.reshape(-1,1),s_w.reshape(-1,1))))
		for col in matrix_df.columns[1:]:
				matrix_df.loc[:,col] = matrix_df.loc[:,col].astype(float)
		matrix_df["Score"] = (matrix_df["S_B"] / (matrix_df["S_W"])).fillna(0)
		matrix_df = matrix_df.sort_values(by="Score",ascending=False).reset_index(drop=True)
		print("Ratios Computed")
		return matrix_df.loc[:,["Feature","Score"]]



