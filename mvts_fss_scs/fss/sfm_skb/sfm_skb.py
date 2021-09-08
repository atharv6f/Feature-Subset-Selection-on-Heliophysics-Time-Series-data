import numpy as np
import pandas as pd
import os
from time import perf_counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from CONSTANTS import RESULTS, DATA_PATH

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
  
def sfm_coef(estimator,X_train, y_train):
    selector = SelectFromModel(estimator)
    selector = selector.fit(X_train, y_train)
    score = np.abs(selector.estimator_.coef_[0])
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": score})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score', ascending=False).reset_index(drop=True)
    return ranked_features


def sfm_fi(estimator,X_train, y_train):
    selector = SelectFromModel(estimator)
    selector = selector.fit(X_train, y_train)
    score = np.abs(selector.estimator_.feature_importances_)
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": score})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score', ascending=False).reset_index(drop=True)
    return ranked_features


def skb(estimator,X_train, y_train):
    selector = SelectKBest(score_func=estimator,k='all').fit(X_train,y_train)
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": selector.scores_})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score',ascending=False).reset_index(drop=True)
    return ranked_features

def main():
	start_time = perf_counter()
	EXPORT_PATH = os.path.join(RESULTS, "ranks")
	
	data = np.load(os.path.join(DATA_PATH, "partition1.npz"), allow_pickle=True)
	X_train = vectorize(data['np_data'])
	y_train = data['target']
	y_train_bin = np.where(y_train=='NF',0,1)

	scaler = MinMaxScaler()
	X_train[:] = scaler.fit_transform(X_train[:])
  
	## SelectKBest
	skb_fval_ranks = skb(f_classif, X_train, y_train)
	skb_fval_ranks.to_csv(EXPORT_PATH+"skb_fval_ranks.csv",index=False)

	skb_mi_ranks = skb(mutual_info_classif, X_train, y_train)
	skb_mi_ranks.to_csv(EXPORT_PATH+"skb_mi_ranks.csv",index=False)

	skb_chi_ranks = skb(chi2, X_train, y_train)
	skb_chi_ranks.to_csv(EXPORT_PATH+"skb_chi_ranks.csv",index=False)

	## SelectFromModel Coef
	logistic = LogisticRegression(solver='liblinear',random_state=777)
	sfm_logistic_ranks = sfm_coef(logistic, X_train, y_train)
	sfm_logistic_ranks.to_csv(EXPORT_PATH+"sfm_logistic_ranks.csv",index=False)

	svc = SVC(kernel='linear')
	sfm_svc_ranks = sfm_coef(svc, X_train, y_train)
	sfm_svc_ranks.to_csv(EXPORT_PATH+"sfm_svc_ranks.csv",index=False)


	## SelectFromModel Feature Imp
	rf = RandomForestClassifier(n_estimators=50)
	sfm_rf_ranks = sfm_fi(rf, X_train, y_train)
	sfm_rf_ranks.to_csv(EXPORT_PATH+"sfm_rf_ranks.csv",index=False)

	xt = ExtraTreesClassifier(n_estimators=50)
	sfm_xt_ranks = sfm_fi(xt, X_train, y_train)
	sfm_xt_ranks.to_csv(EXPORT_PATH+"sfm_xt_ranks.csv",index=False)

	ada = AdaBoostClassifier(n_estimators=50)
	sfm_ada_ranks = sfm_fi(ada, X_train, y_train)
	sfm_ada_ranks.to_csv(EXPORT_PATH+"sfm_ada_ranks.csv",index=False)

	gb = GradientBoostingClassifier(n_estimators=50)
	sfm_gb_ranks = sfm_fi(gb, X_train, y_train)
	sfm_gb_ranks.to_csv(EXPORT_PATH+"sfm_gb_ranks.csv",index=False)

	bag = ExtraTreesClassifier(n_estimators=50)
	sfm_bag_ranks = sfm_fi(bag, X_train, y_train)
	sfm_bag_ranks.to_csv(EXPORT_PATH+"sfm_bag_ranks.csv",index=False)


	time_elapsed = perf_counter() - start_time
	print(f'Total time {time_elapsed} seconds')
    
  
if __name__ == '__main__':
  main()