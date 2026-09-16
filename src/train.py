import os
import json
import logging

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

from src.losses import CategoricalFocalLoss

from src.config import IMG_SIZE, NUM_CLASSES, MODEL_PATH, METRICS_PATH
from src.data_prep import prepare_data, DATA_DIR
from src.preprocessing import prepare_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IN_COLAB = os.path.exists("/content/drive")
BASE_DIR = "/content/drive/MyDrive/diabetic-retinopathy-v2" if IN_COLAB else "."

if IN_COLAB:
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Folder /content/drive terdeteksi tapi Drive belum ter-mount "
            "dengan benar (MyDrive tidak ditemukan). Jalankan di CELL "
            "NOTEBOOK TERPISAH (bukan lewat !python) sebelum training:\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
        )
    logger.info(f"Jalan di Google Colab, BASE_DIR = {BASE_DIR}")
else:
    logger.info("Jalan di lokal (bukan Colab).")


# --- Hyperparameter ---
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 20  # dinaikkan dari 15 -- beri lebih banyak ruang untuk kelas minoritas belajar
FINETUNE_UNFREEZE_RATIO = 0.30  # unfreeze 30% layer teratas EfficientNet
FINETUNE_LR = 1e-5
BATCH_SIZE = 32
FOCAL_LOSS_GAMMA = 2.0  # standar dari paper Focal Loss (Lin et al. 2017)


def make_dataset(df, images_dir, augment: bool, shuffle: bool):
    """
    Buat tf.data.Dataset dari dataframe (kolom id_code, diagnosis).
    Preprocessing (crop_retina + preprocess_input) dipanggil lewat
    prepare_image() dari src/preprocessing.py -- SATU sumber kebenaran
    yang sama dipakai training maupun inference di app.py.
    """

    def _load_and_preprocess(id_code, label):
        path = tf.strings.join([images_dir, "/", id_code])

        def _py_load(path_bytes, label_val):
            path_str = path_bytes.numpy().decode("utf-8")
            img = tf.io.read_file(path_str)
            img = tf.image.decode_png(img, channels=3).numpy()
            processed = prepare_image(img)
            return processed.astype(np.float32), label_val

        processed_img, label_out = tf.py_function(
            _py_load, [path, label], [tf.float32, tf.int64]
        )
        processed_img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        label_out.set_shape([])
        label_one_hot = tf.one_hot(label_out, NUM_CLASSES)
        return processed_img, label_one_hot

    ids = df["id_code"].values
    labels = df["diagnosis"].values.astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((ids, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(1.0),  # rotasi penuh 360 derajat -- retina tidak punya orientasi baku
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ])
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base_model = tf.keras.applications.EfficientNetB3(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # freeze dulu untuk tahap warm-up

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    predictions = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def unfreeze_top_layers(base_model, ratio: float):
    n_layers = len(base_model.layers)
    n_unfreeze = int(n_layers * ratio)
    for layer in base_model.layers[-n_unfreeze:]:
        layer.trainable = True
    logger.info(f"Unfreeze {n_unfreeze}/{n_layers} layer teratas EfficientNetB3.")


def evaluate_on_test(model, test_ds, test_df):
    y_true = test_df["diagnosis"].values
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main():
    data_dir_hint = os.path.join(BASE_DIR, DATA_DIR)
    train_df, val_df, test_df, class_weights, resolved_data_dir = prepare_data(data_dir=data_dir_hint)
    images_dir = os.path.join(resolved_data_dir, "train_images")

    train_ds = make_dataset(train_df, images_dir, augment=True, shuffle=True)
    val_ds = make_dataset(val_df, images_dir, augment=False, shuffle=False)
    test_ds = make_dataset(test_df, images_dir, augment=False, shuffle=False)

    model, base_model = build_model()
    model.compile(optimizer="adam", loss=CategoricalFocalLoss(gamma=FOCAL_LOSS_GAMMA), metrics=["accuracy"])

    checkpoint_path = os.path.join(BASE_DIR, "models", "checkpoint.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=False, save_freq="epoch"
    )

    logger.info(f"=== Warm-up training (head only, {WARMUP_EPOCHS} epochs) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb],
        verbose=2,
    )

    logger.info(f"=== Fine-tuning (unfreeze top {FINETUNE_UNFREEZE_RATIO:.0%}, {FINETUNE_EPOCHS} epochs) ===")
    unfreeze_top_layers(base_model, FINETUNE_UNFREEZE_RATIO)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss=CategoricalFocalLoss(gamma=FOCAL_LOSS_GAMMA),
        metrics=["accuracy"],
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, factor=0.5
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=2,
    )

    logger.info("=== Evaluasi di test set ===")
    metrics = evaluate_on_test(model, test_ds, test_df)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

    model_path = os.path.join(BASE_DIR, MODEL_PATH)
    metrics_path = os.path.join(BASE_DIR, METRICS_PATH)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.save(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model disimpan ke {model_path}")
    logger.info(f"Metrics disimpan ke {metrics_path}")


if __name__ == "__main__":
    main()