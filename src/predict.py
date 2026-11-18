import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_image_array, IMG_SIZE, CLASS_NAMES


def load_trained_model(path):
    import keras
    return keras.models.load_model(str(path))


def predict_image(model, src, threshold=0.5):
    # Same preprocessing as training (shared function) -> sigmoid probability of "Dog"
    arr = load_image_array(src, IMG_SIZE)
    x = arr[None].astype("float32")            # normalization happens inside the model
    p_dog = float(model.predict(x, verbose=0).ravel()[0])
    label = CLASS_NAMES[1] if p_dog >= threshold else CLASS_NAMES[0]
    return dict(label=label, prob_dog=p_dog, prob_cat=1 - p_dog,
                confidence=max(p_dog, 1 - p_dog), image=arr)
