import numpy as np
import cv2
import tensorflow as tf
import matplotlib

from src.config import IMG_SIZE
from src.preprocessing import crop_retina, preprocess_for_model

SUPPORTED_COLORMAPS = {"jet", "viridis", "turbo"}


def get_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Cari layer Conv2D terakhir di model (dari belakang) -- ini yang
    dipakai sebagai basis Grad-CAM, karena feature map di layer ini
    masih menyimpan informasi spasial (di mana letak fitur penting),
    berbeda dengan layer Dense di kepala klasifikasi.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError(
        "Tidak ditemukan layer Conv2D di model ini -- Grad-CAM butuh "
        "minimal satu layer konvolusi untuk menghasilkan heatmap spasial."
    )


def _apply_colormap(heatmap: np.ndarray, colormap: str) -> np.ndarray:
    """
    heatmap: array 2D float, nilai 0-1 (0 = aktivasi rendah, 1 = tinggi).
    Return: array 3D uint8 RGB hasil pewarnaan.
    """
    if colormap not in SUPPORTED_COLORMAPS:
        raise ValueError(
            f"Colormap '{colormap}' tidak didukung. Pilih salah satu: {SUPPORTED_COLORMAPS}"
        )
    cmap = matplotlib.colormaps[colormap]
    colored = cmap(heatmap)[:, :, :3]  # buang alpha channel
    return (colored * 255).astype(np.uint8)


def generate_gradcam(
    model: tf.keras.Model,
    raw_image: np.ndarray,
    class_idx: int = None,
    colormap: str = "jet",
    alpha: float = 0.4,
):
    """
    Hasilkan Grad-CAM overlay untuk satu gambar.

    Parameters
    ----------
    model : model Keras yang sudah di-load (ingat: import src.losses
        dulu sebelum load_model() kalau model pakai CategoricalFocalLoss).
    raw_image : gambar RGB asli (uint8), SEBELUM di-crop/preprocess --
        fungsi ini akan memanggil crop_retina() dan preprocess_for_model()
        sendiri, memakai pipeline yang SAMA PERSIS dengan training
        (satu sumber kebenaran di src/preprocessing.py).
    class_idx : index kelas yang mau divisualisasikan. Kalau None,
        otomatis pakai kelas dengan prediksi tertinggi (prediksi model).
    colormap : "jet", "viridis", atau "turbo".
    alpha : seberapa transparan heatmap saat di-overlay ke gambar asli
        (0 = heatmap tidak terlihat, 1 = heatmap menutup penuh gambar asli).

    Returns
    -------
    dict berisi:
        "overlay": gambar RGB (uint8) hasil overlay heatmap + gambar asli
        "heatmap_raw": heatmap mentah 0-1 (float), ukuran IMG_SIZE
        "class_idx": index kelas yang divisualisasikan (berguna kalau
            caller tidak menentukan class_idx sendiri dan ingin tahu
            kelas mana yang otomatis dipilih)
        "predictions": array probabilitas softmax lengkap (semua kelas)
    """
    cropped = crop_retina(raw_image)
    display_img = cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)
    model_input = preprocess_for_model(cropped)
    batch = np.expand_dims(model_input, axis=0)

    last_conv_layer = get_last_conv_layer(model)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # relu -- hanya pengaruh positif yang relevan

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    heatmap_colored = _apply_colormap(heatmap_resized, colormap)
    overlay = cv2.addWeighted(display_img, 1 - alpha, heatmap_colored, alpha, 0)

    return {
        "overlay": overlay,
        "heatmap_raw": heatmap_resized,
        "class_idx": class_idx,
        "predictions": predictions.numpy()[0],
    }