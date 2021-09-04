import numpy as np
import pandas as pd
import os
from time import perf_counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from mrmr import mrmr_classif

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
  
def mmr_ranking(X_train, y_train):
    selected_features = mrmr_classif(X_train, y_train, K = X_train.shape[1])
    rank_df = pd.DataFrame({"sub_feature" : selected_features, "Rank":list(range(1, 1+len(selected_features)))})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Rank').reset_index(drop=True)
    ranked_features['Score'] = 1/ranked_features['Rank']
    ranked_features.drop('Rank',axis=1,inplace=True)
    return ranked_features


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

	mrmr_ranks = mmr_ranking(X_train, y_train)
	mrmr_ranks.to_csv(EXPORT_PATH+"relief_ranks.csv",index=False)

	time_elapsed = perf_counter() - start_time
	print(f'Total time {time_elapsed} seconds')
    
  
if __name__ == '__main__':
  main()