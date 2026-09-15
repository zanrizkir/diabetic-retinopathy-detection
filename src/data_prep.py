import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import DATA_DIR as DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# Tetap expose DATA_DIR untuk backward-compat dengan modul lain yang
# mengimpornya langsung (misal train.py).
DATA_DIR = DEFAULT_DATA_DIR


def download_dataset() -> str:
    """
    Download dataset APTOS 2019 dari Kaggle via kagglehub.
    Return: path folder hasil download (struktur internal bisa
    bervariasi tergantung sumber dataset).
    """
    import kagglehub

    logger.info("Dataset tidak ditemukan lokal, mengunduh via kagglehub...")
    path = kagglehub.dataset_download("mariaherrerot/aptos2019")
    logger.info(f"Dataset terunduh ke: {path}")
    return path


def _find_data_root(search_path: str) -> str | None:
    """
    Cari folder yang berisi 'train.csv' di dalam search_path (termasuk
    subfolder-nya), karena struktur hasil download bisa nested berbeda-beda.
    Return path folder yang berisi train.csv, atau None kalau tidak ketemu.
    """
    for root, _dirs, files in os.walk(search_path):
        if "train.csv" in files:
            return root
    return None


def resolve_data_dir(data_dir: str = None) -> str:
    """
    Pastikan data_dir yang dipakai benar-benar berisi train.csv.
    - Kalau data_dir diberikan & train.csv ada di situ -> pakai itu.
    - Kalau tidak, coba DEFAULT_DATA_DIR (lokal, dari config.py).
    - Kalau masih tidak ada -> download via kagglehub, lalu cari lokasi
      train.csv di dalam hasil download secara otomatis.

    Ini SATU-SATUNYA tempat logic pencarian path dilakukan -- train.py
    HARUS memanggil ini dan memakai hasilnya, bukan menghitung path sendiri.
    """
    candidates = [data_dir, DEFAULT_DATA_DIR]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "train.csv")):
            logger.info(f"Dataset ditemukan di: {candidate}")
            return candidate

    # Tidak ketemu di kandidat manapun -> download
    downloaded_path = download_dataset()
    found_root = _find_data_root(downloaded_path)
    if found_root is None:
        raise FileNotFoundError(
            f"train.csv tidak ditemukan di manapun, termasuk di dalam hasil "
            f"download ({downloaded_path}). Cek struktur dataset secara manual."
        )
    logger.info(f"Dataset (hasil download) ditemukan di: {found_root}")
    return found_root


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load train.csv dari data_dir yang SUDAH di-resolve (lihat resolve_data_dir).
    """
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tidak ditemukan {csv_path}.")

    df = pd.read_csv(csv_path)
    df["id_code"] = df["id_code"].astype(str) + ".png"
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Split df menjadi train/val/test dengan proporsi (1 - test_size - val_size)
    / val_size / test_size, stratified berdasarkan kolom 'diagnosis'.
    Default 70/15/15.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["diagnosis"], random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["diagnosis"],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(df: pd.DataFrame) -> dict:
    classes = np.array(sorted(df["diagnosis"].unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["diagnosis"])
    return {int(c): float(w) for c, w in zip(classes, weights)}


def print_split_summary(name: str, split_df: pd.DataFrame, class_weights: dict = None):
    print(f"{name}: {len(split_df)} sample")
    counts = split_df["diagnosis"].value_counts().sort_index()
    for cls, count in counts.items():
        w_str = f", weight {class_weights[cls]:.3f}" if class_weights else ""
        print(f"  class {cls}: {count} sample{w_str}")


def prepare_data(
    data_dir: str = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Fungsi utama: resolve lokasi data, load metadata, split, hitung class weight.

    Parameters
    ----------
    data_dir : str, optional
        Path folder yang diharapkan berisi train.csv & train_images/.
        Kalau None, pakai DEFAULT_DATA_DIR dari config.py (untuk run lokal).
        Kalau tidak ditemukan di sini, otomatis fallback ke download.

    Return
    ------
    train_df, val_df, test_df, class_weights, resolved_data_dir
        `resolved_data_dir` WAJIB dipakai pemanggil (train.py) untuk
        menyusun path ke folder train_images/, supaya konsisten dengan
        lokasi yang benar-benar dipakai di sini.
    """
    resolved_data_dir = resolve_data_dir(data_dir)

    df = load_metadata(resolved_data_dir)
    print(f"Total dataset: {len(df)} gambar, {df['diagnosis'].nunique()} kelas\n")

    train_df, val_df, test_df = stratified_split(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    class_weights = compute_class_weights(train_df)

    print("=== Ringkasan Split ===")
    print_split_summary("Train", train_df, class_weights)
    print_split_summary("Validation", val_df)
    print_split_summary("Test", test_df)

    return train_df, val_df, test_df, class_weights, resolved_data_dir


if __name__ == "__main__":
    train_df, val_df, test_df, class_weights, resolved_data_dir = prepare_data()

    train_df.to_csv(os.path.join(resolved_data_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(resolved_data_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(resolved_data_dir, "test_split.csv"), index=False)
    print(f"\nSplit CSV disimpan di {resolved_data_dir}/train_split.csv, val_split.csv, test_split.csv")