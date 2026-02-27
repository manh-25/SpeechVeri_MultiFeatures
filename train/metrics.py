import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def compute_mindcf(y_true, scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    return np.min(dcf), thresholds[np.argmin(dcf)]