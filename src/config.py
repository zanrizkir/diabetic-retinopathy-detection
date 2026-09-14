"""
config.py — Konfigurasi global untuk project Diabetic Retinopathy Detection v2
"""

# Ukuran input gambar untuk model EfficientNetB3
IMG_SIZE = (300, 300)

# Label kelas (sesuai encoding dataset APTOS 2019)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Jumlah kelas output
NUM_CLASSES = 5

# Path dataset APTOS 2019 (setelah download via kagglehub)
DATA_DIR = "data/aptos2019-blindness-detection"

# Path untuk menyimpan model hasil training
MODEL_PATH = "models/efficientnetb3_dr.keras"

# Path untuk menyimpan metrics hasil evaluasi
METRICS_PATH = "models/metrics.json"
