# from mvts_fss_scs.fss.base_fss import BaseFSS
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import RFE
from time import perf_counter
from sklearn.preprocessing import MinMaxScaler



# class RFE(BaseFSS):
#   def __init__(self,data = None,**kwargs):
	
#     if data:
#       self.data = data
#     else:
#       super().__init__(data)
		
#     """
#     1. This is where use initialize the keyword arguments you need for your algorithm.
#     Refer pie.py or csfs.py to get a rough idea of the implementation.

#     2. You need to implement a method called rank which returns your scores (dataframe) in desceneding order
#       in the following format.

#       Features | Score

#     3. Create an instance of the above class in the main __init__file
#     """
#   def rank(self):
#     pass

def rfe(estimator, X_train, y_train):
		selector = RFE(estimator, n_features_to_select=1, step=2)   
		selector.fit(X_train, y_train)
		rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Rank": selector.ranking_})
		rank_df['feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
		ranked_features = rank_df.groupby('feature',as_index=False).agg('mean').sort_values('Rank').reset_index(drop=True)
		ranked_features['Score'] = 1/ranked_features['Rank']
		ranked_features.drop('Rank',axis=1,inplace=True)
		return ranked_features

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
	EXPORT_PATH = "Results/ranks2/"

	data = np.load(os.path.join(DATA_PATH, "partition1.npz"), allow_pickle=True)
	X_train = vectorize(data['np_data'])
	y_train = data['target']

	scaler = MinMaxScaler()
	X_train[:] = scaler.fit_transform(X_train[:])
	
	logistic = LogisticRegression(solver='liblinear',random_state=777)
	rfe_logistic_ranks = rfe(logistic, X_train, y_train)
	rfe_logistic_ranks.to_csv(EXPORT_PATH+"rfe_logistic_ranks.csv",index=False)

	svc = SVC(kernel='linear')
	rfe_svc_ranks = rfe(svc, X_train, y_train)
	rfe_svc_ranks.to_csv(EXPORT_PATH+"rfe_svc_ranks.csv",index=False)

	rf = RandomForestClassifier(n_estimators=50)
	rfe_rf_ranks = rfe(rf, X_train, y_train)
	rfe_rf_ranks.to_csv(EXPORT_PATH+"rfe_rf_ranks.csv",index=False)

	xt = ExtraTreesClassifier(n_estimators=50)
	rfe_xt_ranks = rfe(xt, X_train, y_train)
	rfe_xt_ranks.to_csv(EXPORT_PATH+"rfe_xt_ranks.csv",index=False)

	ada = AdaBoostClassifier(n_estimators=50)
	rfe_ada_ranks = rfe(ada, X_train, y_train)
	rfe_ada_ranks.to_csv(EXPORT_PATH+"rfe_ada_ranks.csv",index=False)

	gb = GradientBoostingClassifier(n_estimators=50)
	rfe_gb_ranks = rfe(gb, X_train, y_train)
	rfe_gb_ranks.to_csv(EXPORT_PATH+"rfe_gb_ranks.csv",index=False)

	bag = ExtraTreesClassifier(n_estimators=50)
	rfe_bag_ranks = rfe(bag, X_train, y_train)
	rfe_bag_ranks.to_csv(EXPORT_PATH+"rfe_bag_ranks.csv",index=False)
	
	time_elapsed = perf_counter() - start_time
	print(f'Total time {time_elapsed} seconds')
		
	
if __name__ == '__main__':
	main()
