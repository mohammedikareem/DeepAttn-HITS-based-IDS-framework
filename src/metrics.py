
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)

def calc_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_ = roc_auc_score(y_true, y_proba)
    except Exception:
        auc_ = float('nan')
    ap   = average_precision_score(y_true, y_proba)
    mcc  = matthews_corrcoef(y_true, y_pred)
    bal  = balanced_accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = np.sqrt(max(rec, 0.0) * max(spec, 0.0))
    return acc, prec, rec, f1, auc_, mcc, bal, spec, gmean, ap, (tn, fp, fn, tp)
