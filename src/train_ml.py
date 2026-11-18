import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def compute_metrics(y, pred, score):
    # Positive class = Dog (1)
    return dict(Accuracy=accuracy_score(y, pred),
                Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0),
                ROC_AUC=roc_auc_score(y, score))


def stratified_subsample(y, n, seed=42):
    # indices of a class-balanced-by-proportion subset of size n
    y = np.asarray(y)
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = int(round(n * len(ci) / len(y)))
        idx.extend(rng.choice(ci, size=min(k, len(ci)), replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def sk_scores(model, X):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
    else:
        score = pred.astype(float)
    return pred, score


def timed_fit(model, X, y):
    t0 = time.time()
    model.fit(X, y)
    return model, time.time() - t0
