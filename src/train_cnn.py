import time
import keras
from keras import layers

SEED = 42

# Three experimental CNN designs (Q46). "A" is the baseline architecture from the project brief.
ARCHS = {
    "CNN-A (3 conv, 3x3, drop 0.4)": dict(filters=(32, 64, 128), kernel=3, dense=128, dropout=0.4, bn=False, gap=False),
    "CNN-B (4 conv, BN, GAP, drop 0.5)": dict(filters=(32, 64, 128, 256), kernel=3, dense=128, dropout=0.5, bn=True, gap=True),
    "CNN-C (3 conv, 5x5, small, drop 0.3)": dict(filters=(16, 32, 64), kernel=5, dense=64, dropout=0.3, bn=False, gap=False),
}


def augmentation_layers():
    # Active only when training=True, so validation / test / inference are never augmented
    aug = [layers.RandomFlip("horizontal"),
           layers.RandomRotation(0.06),
           layers.RandomZoom(0.15),
           layers.RandomTranslation(0.1, 0.1)]
    try:
        aug.append(layers.RandomShear(x_factor=0.1, y_factor=0.1))
    except Exception:
        pass
    return keras.Sequential(aug, name="augmentation")


def build_cnn(cfg, input_shape=(128, 128, 3), augment=False, name="cnn"):
    inp = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inp)           # pixel normalization 0-255 -> 0-1 lives inside the model
    if augment:
        x = augmentation_layers()(x)
    for f in cfg["filters"]:
        x = layers.Conv2D(f, cfg["kernel"], padding="same", use_bias=not cfg["bn"])(x)
        if cfg["bn"]:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x) if cfg["gap"] else layers.Flatten()(x)
    x = layers.Dense(cfg["dense"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=name)


def build_transfer_model(pretrained="imagenet", input_shape=(128, 128, 3), dropout=0.3):
    base = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=pretrained)
    base.trainable = False                         # frozen backbone for the first baseline
    inp = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="mobilenetv2_transfer"), base


def unfreeze_top_layers(base, n=30):
    base.trainable = True
    for l in base.layers[:-n]:
        l.trainable = False
    for l in base.layers:                           # keep BatchNorm statistics frozen
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def compile_and_fit(model, name, X_tr, y_tr, X_va, y_va, model_dir, epochs=20,
                    lr=1e-3, patience=4, batch_size=64):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    safe = "".join(c if c.isalnum() else "_" for c in name)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(f"{model_dir}/{safe}.keras", monitor="val_loss", save_best_only=True),
           keras.callbacks.CSVLogger(f"{model_dir}/{safe}_log.csv")]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=batch_size, callbacks=cbs, shuffle=True, verbose=2)
    return hist, time.time() - t0
