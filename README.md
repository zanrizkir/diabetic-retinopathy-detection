# Diabetic Retinopathy Detection v2

Sistem klasifikasi **Diabetic Retinopathy (DR)** dari citra fundus retina menggunakan deep learning berbasis **EfficientNetB3**, dilengkapi dengan:

- 🔬 **Klasifikasi 5 tingkat keparahan DR** — No DR, Mild, Moderate, Severe, Proliferative DR
- 🗺️ **Grad-CAM visualization** — Peta aktivasi untuk interpretabilitas prediksi model
- 🤖 **AI Clinical Intelligence** — Rekomendasi klinis otomatis via **Google Gemini API**
- 📊 **Dashboard interaktif** — Antarmuka berbasis **Streamlit** untuk upload & analisis gambar

## Dataset

Menggunakan dataset [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dari Kaggle.

## Arsitektur Model

- **Backbone**: EfficientNetB3 (pretrained ImageNet)
- **Input size**: 300×300 px
- **Output**: 5 kelas (softmax)
- **Transfer learning** dengan fine-tuning bertahap

---

## Setup

> ⚠️ Section ini akan dilengkapi secara bertahap di setiap tahap pengembangan.

### 1. Clone repository & persiapkan environment

```bash
# Clone repo
git clone <repo-url>
cd diabetic-retinopathy-v2

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi environment

```bash
cp .env.example .env
# Edit .env dan isi GEMINI_API_KEY dengan API key Anda
```

### 3. Persiapan data

> *(Akan diisi di tahap data_prep)*

### 4. Training model

> *(Akan diisi di tahap train)*

### 5. Menjalankan aplikasi

> *(Akan diisi di tahap app)*

---

## Struktur Project

```
diabetic-retinopathy-v2/
├── src/
│   ├── config.py                   # Konfigurasi global
│   ├── data_prep.py                # Download & persiapan dataset
│   ├── preprocessing.py            # Augmentasi & preprocessing citra
│   ├── train.py                    # Training pipeline EfficientNetB3
│   ├── gradcam.py                  # Grad-CAM visualization
│   └── clinical_intelligence.py   # Integrasi Gemini API
├── data/                           # Dataset (tidak di-commit ke git)
├── models/                         # Model tersimpan (tidak di-commit ke git)
├── static/                         # Asset CSS untuk Streamlit
├── tests/                          # Unit tests
├── app.py                          # Aplikasi Streamlit utama
└── requirements.txt
```

## Lisensi

MIT License
