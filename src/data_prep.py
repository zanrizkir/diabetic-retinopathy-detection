import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "train_images")


def download_dataset() -> None:
    """
    Opsional: download dataset APTOS 2019 dari Kaggle via kagglehub,
    kalau belum ada secara lokal. Butuh kaggle.json terpasang
    (~/.kaggle/kaggle.json) atau environment variable KAGGLE_KEY/KAGGLE_USERNAME.
    """
    import kagglehub

    logger.info("Dataset tidak ditemukan lokal, mengunduh via kagglehub...")
    path = kagglehub.dataset_download("mariaherrerot/aptos2019")
    logger.info(f"Dataset terunduh ke: {path}")
    return path


def load_metadata() -> pd.DataFrame:
    """
    Load train.csv, tambahkan ekstensi .png ke id_code, dan pastikan
    kolom diagnosis bertipe int (0-4).
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Tidak ditemukan {CSV_PATH}. Pastikan dataset sudah di-copy "
            f"ke folder {DATA_DIR}/, atau panggil download_dataset() dulu."
        )

    df = pd.read_csv(CSV_PATH)
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
    # Tahap 1: pisahkan test set dulu
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["diagnosis"],
        random_state=random_state,
    )

    # Tahap 2: dari sisa train_val, pisahkan val set
    # val_size dihitung ulang relatif terhadap train_val_df supaya proporsi
    # akhir terhadap df ASLI tetap sesuai target (val_size dari total).
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
    """
    Hitung class_weight (strategi 'balanced') dari kolom 'diagnosis'
    pada dataframe (biasanya dipanggil dengan train_df saja, BUKAN
    keseluruhan dataset).
    """
    classes = np.array(sorted(df["diagnosis"].unique()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=df["diagnosis"]
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def print_split_summary(name: str, split_df: pd.DataFrame, class_weights: dict = None):
    """Cetak ringkasan jumlah sample & distribusi kelas per split."""
    print(f"{name}: {len(split_df)} sample")
    counts = split_df["diagnosis"].value_counts().sort_index()
    for cls, count in counts.items():
        w_str = f", weight {class_weights[cls]:.3f}" if class_weights else ""
        print(f"  class {cls}: {count} sample{w_str}")


def prepare_data(test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Fungsi utama: load metadata, split, hitung class weight.
    Return: train_df, val_df, test_df, class_weights (dict)
    """
    if not os.path.exists(CSV_PATH):
        download_dataset()

    df = load_metadata()
    print(f"Total dataset: {len(df)} gambar, {df['diagnosis'].nunique()} kelas\n")

    train_df, val_df, test_df = stratified_split(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    class_weights = compute_class_weights(train_df)

    print("=== Ringkasan Split ===")
    print_split_summary("Train", train_df, class_weights)
    print_split_summary("Validation", val_df)
    print_split_summary("Test", test_df)

    return train_df, val_df, test_df, class_weights


if __name__ == "__main__":
    train_df, val_df, test_df, class_weights = prepare_data()

    # Simpan hasil split ke CSV supaya reproducible (train.py tinggal load ini)
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)
    print(f"\nSplit CSV disimpan di {DATA_DIR}/train_split.csv, val_split.csv, test_split.csv")