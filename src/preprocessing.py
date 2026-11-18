# Shared image preprocessing: used by training notebooks AND the deployed app,
# so training-time and inference-time preprocessing are identical.
import io
import numpy as np
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

IMG_SIZE = 128          # CNN / transfer-learning input size
ML_SIZE = 64            # reduced size for traditional ML (64x64x3 = 12288 features)
CLASS_NAMES = ["Cat", "Dog"]
LABEL_MAP = {"Cat": 0, "Dog": 1}


class InvalidImageError(Exception):
    pass


def load_image_array(src, size=IMG_SIZE):
    # Open -> fix EXIF rotation -> force RGB (handles gray/RGBA/palette) -> resize -> uint8 array
    try:
        if hasattr(src, "read"):
            src = io.BytesIO(src.read())
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im = im.resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception as e:
        raise InvalidImageError(f"Cannot read image: {type(e).__name__}: {e}")


def _safe_load(path, size):
    return load_image_array(path, size)


def load_dataset_array(paths, size=IMG_SIZE, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(lambda p: _safe_load(p, size), paths))
    return np.stack(arrs).astype(np.uint8)


def downsample_2x(X):
    # exact 2x2 average-pooling (e.g. 128 -> 64), done in chunks to save memory
    n, h, w, c = X.shape
    out = np.empty((n, h // 2, w // 2, c), dtype=np.uint8)
    for i in range(0, n, 2000):
        b = X[i:i + 2000].astype(np.float32)
        b = b.reshape(len(b), h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))
        out[i:i + 2000] = np.round(b).astype(np.uint8)
    return out


def normalize(X):
    # pixel range 0-255 -> 0-1
    return X.astype(np.float32) / 255.0
