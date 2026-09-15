import numpy as np
import cv2
import logging

from src.config import IMG_SIZE

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:  # TensorFlow belum terpasang saat development awal
    preprocess_input = None
    logger.warning(
        "TensorFlow belum terpasang -- preprocess_input tidak tersedia. "
        "Fungsi preprocess_for_model() akan gagal sampai TensorFlow di-install."
    )


def crop_retina(image: np.ndarray, min_area_ratio: float = 0.10) -> np.ndarray:
    """
    Deteksi kontur retina (objek terbesar/paling terang di gambar fundus)
    dan crop gambar menjadi persegi (1:1) berpusat di kontur tersebut.

    Parameters
    ----------
    image : np.ndarray
        Gambar RGB, shape (H, W, 3), dtype uint8.
    min_area_ratio : float
        Ambang minimum rasio (luas kontur terdeteksi / luas gambar total).
        Kalau kontur yang ditemukan lebih kecil dari ini, dianggap deteksi
        gagal/tidak reliable -> fallback ke gambar original.

    Returns
    -------
    np.ndarray
        Gambar hasil crop, MASIH dalam ukuran aslinya (belum di-resize
        ke IMG_SIZE -- itu tanggung jawab preprocess_for_model()).
    """
    h, w = image.shape[:2]
    total_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold Otsu: pisahkan area terang (retina) dari background gelap.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.info("[crop_retina] FALLBACK - tidak ada kontur ditemukan, pakai gambar original.")
        return image

    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)

    if contour_area < min_area_ratio * total_area:
        logger.info(
            "[crop_retina] FALLBACK - kontur terdeteksi terlalu kecil "
            f"({contour_area / total_area:.1%} dari total gambar), pakai gambar original."
        )
        return image

    x, y, bw, bh = cv2.boundingRect(largest)

    # --- Paksa aspect ratio 1:1 ---
    side = max(bw, bh)
    center_x, center_y = x + bw // 2, y + bh // 2

    x1 = center_x - side // 2
    y1 = center_y - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # Geser dulu (bukan potong) selama gambar asli cukup besar untuk
    # menampung `side` penuh di masing-masing sumbu.
    if x1 < 0 and (x2 - x1) <= w:
        x2 -= x1
        x1 = 0
    if y1 < 0 and (y2 - y1) <= h:
        y2 -= y1
        y1 = 0
    if x2 > w and (x2 - x1) <= w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h and (y2 - y1) <= h:
        y1 -= (y2 - h)
        y2 = h

    # Hitung area yang benar-benar tersedia di gambar sumber (bisa lebih
    # kecil dari `side` kalau gambar sumber sendiri lebih sempit dari
    # ukuran crop yang dibutuhkan).
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w, x2), min(h, y2)
    partial = image[src_y1:src_y2, src_x1:src_x2]

    # Padding hitam simetris untuk menutup kekurangan sisi mana pun,
    # supaya output DIJAMIN selalu side x side (bukan cuma "area maksimal
    # yang tersedia" seperti sebelumnya).
    pad_left = src_x1 - x1
    pad_top = src_y1 - y1
    pad_right = x2 - src_x2
    pad_bottom = y2 - src_y2

    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = cv2.copyMakeBorder(
            partial, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        logger.info(
            f"[crop_retina] Sumber lebih sempit dari sisi crop, padding hitam "
            f"ditambahkan (top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right})."
        )
    else:
        cropped = partial

    actual_h, actual_w = cropped.shape[:2]
    aspect = actual_w / actual_h if actual_h else 0
    logger.info(
        f"[crop_retina] BERHASIL - bbox asli: {bw}x{bh}, "
        f"hasil crop: {actual_w}x{actual_h} (aspect: {aspect:.3f})"
    )

    assert actual_w == actual_h == side, "crop_retina harus selalu menghasilkan persegi side x side"
    return cropped


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Resize ke IMG_SIZE dan terapkan preprocessing resmi EfficientNet.
    TIDAK ADA pembagian manual /255.0 di sini.
    """
    if preprocess_input is None:
        raise RuntimeError(
            "TensorFlow belum terpasang. Install dengan: pip install tensorflow"
        )

    resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32)
    return preprocess_input(array)


def prepare_image(image: np.ndarray) -> np.ndarray:
    """
    Fungsi gabungan: crop_retina() -> preprocess_for_model().
    INI SATU-SATUNYA fungsi yang dipakai baik oleh train.py maupun
    app.py, supaya tidak ada duplikasi logic preprocessing.
    """
    cropped = crop_retina(image)
    return preprocess_for_model(cropped)