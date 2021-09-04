import numpy as np
import pandas as pd
import os
from time import perf_counter
from skrebate import ReliefF
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis


def vectorize(data_3d):
	first24 = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
	'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM',
	'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
	'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE']
	header = []
	data_min = np.min(data_3d, axis=1)

	header.extend([x+'_min' for x in first24])
	data_max = np.max(data_3d, axis=1)
	
	header.extend([x+'_max' for x in first24])
	data_median = np.median(data_3d, axis=1)
	
	header.extend([x+'_median' for x in first24])
	data_sd = np.std(data_3d, axis=1)
	header.extend([x+'_sd' for x in first24])
	data_last = data_3d[:,-1,:]
	header.extend([x+'_last' for x in first24])
	data_skew = skew(data_3d, axis=1)
	header.extend([x+'_skew' for x in first24])
	data_kurtosis = kurtosis(data_3d, axis=1)
	header.extend([x+'_kurtosis' for x in first24])
	combined_data = np.hstack((data_min, data_max, data_median, data_sd, data_last, data_skew, data_kurtosis))
	combined_df = pd.DataFrame(combined_data, columns=header)
	return combined_df
  

def main():
	start_time = perf_counter()
	DATA_PATH = "mvts_fss_scs/preprocessed_data"
	EXPORT_PATH = "Results/ranks/"
	
	data = np.load(os.path.join(DATA_PATH, "partition1.npz"), allow_pickle=True)
	X_train = vectorize(data['np_data'])
	y_train = data['target']
	y_train_bin = np.where(y_train=='NF',0,1)

	scaler = MinMaxScaler()
	X_train[:] = scaler.fit_transform(X_train[:])

	def relief_ranking(X_train, y_train):
		X_train_values, y_train_values = X_train.values, y_train
		selector = ReliefF(n_features_to_select=X_train.shape[1], n_neighbors=0.01, n_jobs=12)
		selector.fit(X_train_values, y_train_values)
		rank_df = pd.DataFrame({"sub_feature" : X_train.columns[selector.top_features_], "rank":list(range(1, 1+X_train.shape[1]))})
		rank_df['feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
		ranked_features = rank_df.groupby('feature',as_index=False).agg('mean').sort_values('rank').reset_index(drop=True)
		ranked_features['Score'] = 1/ranked_features['rank']
		ranked_features.drop('rank',axis=1,inplace=True)
		return ranked_features

	relief_ranks = relief_ranking(X_train, y_train)
	relief_ranks.to_csv(EXPORT_PATH+"relief_ranks.csv",index=False)

	time_elapsed = perf_counter() - start_time
	print(f'Total time {time_elapsed} seconds')
    
  
if __name__ == '__main__':
  main()
