import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_score(loss, y_true, _type, ratio=0.2):
    per = np.percentile(loss, 100-ratio*100)
    y_pred = np.zeros_like(loss)
    y_pred[loss < per] = 0
    y_pred[loss >= per] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    print("%s: Prec = %.4f | Rec = %.4f | F1 = %.4f " % (_type, precision, recall, f1))
    return precision, recall, f1