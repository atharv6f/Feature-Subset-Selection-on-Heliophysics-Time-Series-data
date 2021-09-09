from sklearn.metrics import confusion_matrix


def calc_tss(y_true=None, y_predict=None):
    """
    Calculates the true skill score for binary classification based on the output of the confusion
    table function
    """
    scores = confusion_matrix(y_true, y_predict).ravel()
    TN, FP, FN, TP = scores
    tp_rate = TP / float(TP + FN) if TP > 0 else 0  
    fp_rate = FP / float(FP + TN) if FP > 0 else 0
    
    return tp_rate - fp_rate


def calc_hss2(y_true,y_pred):
    """
    :return: Heidke skill score. Following sklearn's implementation of other metrics, when
    the denominator is zero, it returns zero.
    """
    scores = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = scores

    numer = 2 * ((tp * tn) - (fn * fp))
    denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = (numer / float(denom)) if denom != 0 else 0
    return hss
