import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


def dists(pd_series: pd.Series) -> pd.Series:
    """
    This function returns discrete distribution.
    prob = frequency / total elements

    :param pd_series: Input of pandas series
    :return: Returns discrete distribution
    """
    freq_prob = pd.Series(pd_series.value_counts().div(len(pd_series)), name= pd_series.name)
    return freq_prob

def entropy_X(pd_series: pd.Series,
                base: float = 2) -> float:
    """
    This function returns entropy of the given pandas series.
    Base is assumed to be 2 but can be changed.
    
    :param pd_series: Input pandas series
    :param base: Input for the base of the log
    :return: Returns float entropy value
    """

    return entropy(dists(pd_series), base = base)

def entropy_XY(X: pd.Series,
         Y: pd.Series,
         base: float = 2) -> float:
    """
    This function returns entropy of the given feature w.r.t. class.
    Red. for conditional probability code:
    https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python
    Base is assumed to be 2 but can be changed.
    
    :param X: Input feature as pandas series
    :param Y: Input class as pandas series
    :param base: Input for the base of the log
    :return: Returns float entropy value
    """
    # Y probability
    Y_d = dists(Y)
    temp_data = {X.name: X.to_list(), Y.name: Y.to_list()}
    df = pd.DataFrame.from_dict(temp_data)

    # conditional probability
    con_prob_num = df.groupby([X.name, Y.name]).size().div(len(df))
    con_prob = con_prob_num.div(Y_d, axis=0, level=Y.name).swaplevel()

    # Entropy X|Y
    entr_XY = []

    for val in Y_d.index.to_list():
        entr_XY.append(entropy(con_prob[val], base= base))
    
    # Conditional Entropy H(X|Y)
    H_XY = Y_d.dot(entr_XY)

    return H_XY

def information_gain_XY(X: pd.Series,
                        Y: pd.Series, 
                        base: float = 2) -> float:
    """
    This function returns information gain of the given feature w.r.t. class.
    Base is assumed to be 2 but can be changed. 
    
    Information Gain = H(X) - H(X|Y)    
    
    :param X: Input feature as pandas series
    :param Y: Input class as pandas series
    :param base: Input for the base of the log
    :return: Returns float information gain
    """
    
    return entropy_X(X, base) - entropy_XY(X, Y, base)

def symmetrical_uncertainty(X: pd.Series,
                            Y: pd.Series,
                            base: float = 2) -> float:
    """
    This function returns symmetrical uncertainty of the given feature w.r.t. class.
    Base is assumed to be 2 but can be changed.
    
    Symmetrical Uncertainty = 2 * information gain / (H(X) + H(Y))

    :param X: Input feature as pandas series
    :param Y: Input class as pandas series
    :param base: Input for the base of the log
    :return: Returns float entropy value
    """
    #X_n = X.to_numpy()
    #Y_n = Y.to_numpy()
    su = 2 * information_gain_XY(X, Y, base) / (entropy_X(X, base) + entropy_X(Y, base))
    #su = 2 * mutual_info(X_n, Y_n) / (entropy_X(X) + entropy_X(Y))
    return su
