# nlp-labeling-attraction-reviews

Repositori ini memuat dataset, kode fungsi, dan dokumentasi pendukung untuk
penelitian tugas akhir berjudul **"Implementasi Natural Language Processing
untuk Labeling Dataset Ulasan Attraction di Indonesia"** (Universitas Bakrie).

Penelitian ini membangun dataset ulasan *attraction* di Bali yang berlabel
sentimen dan multibahasa, kemudian mengevaluasi secara komparatif lima
pendekatan model NLP untuk klasifikasi sentimen guna menentukan model yang
paling efektif untuk pelabelan dataset akhir.

---

## Deskripsi Dataset

Dataset dihimpun dari dua platform ulasan, yaitu **TripAdvisor** dan
**Google Review**, dan mencakup tiga bahasa: Bahasa Inggris (`en`),
Bahasa Indonesia (`id`), dan Bahasa Spanyol (`es`).

| Tahap | Jumlah |
|-------|--------|
| Data mentah (TripAdvisor 6.768 + Google Review 15.360) | 22.128 baris |
| Dataset final setelah pembersihan & prapemrosesan | 10.777 baris (13 kolom) |
| Pembagian train / validation / test (stratified 70:15:15) | 7.543 / 1.617 / 1.617 |

Berkas dataset final berlabel tersedia pada folder [`data/`](data/).

### Skema Proxy Label

Label acuan (`proxy_sentiment`) dibentuk melalui pemetaan rating pengguna:

| Rating | Proxy Sentiment |
|--------|-----------------|
| 1 – 2  | negatif |
| 3      | netral  |
| 4 – 5  | positif |

Proxy label diperlakukan sebagai **distant supervision label (silver label)**,
bukan sebagai *gold standard* maupun hasil anotasi manual manusia. Label
sentimen akhir pada dataset merupakan hasil pelabelan otomatis oleh model
terpilih, bukan anotasi manusia.

---

## Model yang Dievaluasi

Lima pendekatan dievaluasi pada test set yang sama, dengan **macro F1-score**
sebagai metrik utama:

| Pendekatan | Macro F1 |
|------------|----------|
| GPT prompt-based zero-shot | 70,88% |
| mBERT-XNLI zero-shot | 46,97% |
| XLM-RoBERTa zero-shot | 59,05% |
| mBERT fine-tuned | 66,39% |
| **XLM-RoBERTa fine-tuned** (model terpilih) | 70,95% |

Model **XLM-RoBERTa fine-tuned** (accuracy 89,98%; weighted F1 89,68%)
ditetapkan sebagai model untuk pelabelan sentimen pada dataset akhir. Dataset
akhir berlabel memiliki tingkat kesesuaian 93,28% dengan proxy label.

---

## Struktur Repositori

```
nlp-labeling-attraction-reviews/
├── data/                 # Dataset final berlabel (10.777 baris)
├── src/                  # Fungsi-fungsi inti tiap tahap penelitian
│   ├── cleaning.py       # Tahap 1: pembersihan data & verifikasi bahasa
│   ├── preprocessing.py  # Tahap 2: prapemrosesan, proxy label, split dataset
│   └── modeling.py       # Tahap 3: evaluasi lima model & pelabelan dataset
├── requirements.txt      # Daftar dependensi
└── README.md
```

> Catatan: berkas pada `src/` merupakan ekstraksi fungsi inti dari tiga
> notebook penelitian. Sel eksplorasi, pencetakan progres, dan visualisasi
> tidak disertakan agar kode lebih ringkas dan mudah dibaca.

---

## Alur Penelitian

Kode disusun mengikuti tiga tahap yang berurutan:

1. **`cleaning.py`** — Standardisasi kolom, pembersihan bertahap (hapus
   ulasan kosong, duplikat, dan ulasan terlalu pendek), penyaringan bahasa
   berdasarkan label platform, verifikasi bahasa dengan `langdetect`, serta
   penyaringan akhir berdasarkan hasil deteksi.

2. **`preprocessing.py`** — Validasi data, prapemrosesan ringan (tanpa
   stopword removal, stemming, atau lemmatization), pembentukan proxy label
   dari rating, penghapusan duplikat akhir, dan pembagian dataset 70:15:15
   secara *stratified* berdasarkan kombinasi bahasa × sentimen.

3. **`modeling.py`** — Evaluasi komparatif lima pendekatan model (dua
   zero-shot NLI, GPT prompt-based, dan dua fine-tuning), evaluasi per bahasa,
   serta pelabelan seluruh dataset menggunakan model terpilih.

---

## Cara Menggunakan

```bash
# 1. Klon repositori
git clone https://github.com/nessfs/nlp-labeling-attraction-reviews.git
cd nlp-labeling-attraction-reviews

# 2. Pasang dependensi
pip install -r requirements.txt
```

Fungsi-fungsi pada `src/` dapat diimpor sesuai tahap yang dibutuhkan,
misalnya:

```python
from src.preprocessing import light_text_preprocessing, rating_to_sentiment

teks_bersih = light_text_preprocessing(teks_mentah)
label       = rating_to_sentiment(4)   # -> "positif"
```

Seluruh eksperimen menggunakan `random_state = 42` untuk memastikan
reproduktibilitas.

---

## Catatan

- Dataset ini dihimpun untuk keperluan penelitian akademik.
- Kolom `user_id` tidak disertakan pada dataset yang dipublikasikan demi
  menjaga privasi pengguna.
- Fine-tuning dijalankan pada lingkungan ber-GPU (Google Colab).
