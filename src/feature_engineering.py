import numpy as np
from joblib import Parallel, delayed


def flatten(X_float):
    return X_float.reshape(len(X_float), -1)


def to_grayscale(X_uint8):
    # ITU-R 601 luma, returns float32 (N, H, W) in [0, 1]
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (X_uint8.astype(np.float32) @ w) / 255.0


def hog_features(X_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), n_jobs=-1):
    from skimage.feature import hog
    feats = Parallel(n_jobs=n_jobs)(
        delayed(hog)(g, orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, block_norm="L2-Hys") for g in X_gray)
    return np.asarray(feats, dtype=np.float32)
