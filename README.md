# nlp-labeling-attraction-reviews

Repositori ini memuat dataset, kode, dan dokumentasi pendukung untuk tugas
akhir berjudul **"Implementasi Natural Language Processing untuk Labeling
Dataset Ulasan Attraction di Indonesia"** (Universitas Bakrie).

Penelitian ini membangun dataset ulasan *attraction* di Bali yang berlabel
sentimen dan multibahasa, kemudian mengevaluasi secara komparatif lima
pendekatan model NLP untuk klasifikasi sentimen guna menentukan model yang
paling efektif untuk pelabelan dataset akhir. Dataset akhir yang telah
dilabeli merupakan salah satu output utama penelitian ini.

---

## Deskripsi Dataset

Dataset dihimpun dari dua platform ulasan, yaitu **TripAdvisor** dan
**Google Review**, dan mencakup tiga bahasa: Bahasa Inggris (`en`),
Bahasa Indonesia (`id`), dan Bahasa Spanyol (`es`).

| Tahap | Jumlah |
|-------|--------|
| Raw Data (TripAdvisor 6.768 + Google Review 15.360) | 22.128 baris |
| Dataset final setelah pembersihan & prapemrosesan | 10.777 baris |
| Pembagian train / validation / test (*stratified* 70:15:15) | 7.543 / 1.617 / 1.617 |

Berkas dataset final berlabel tersedia pada folder [`data/`](data/).

### Skema Kolom Dataset Final

Versi yang dipublikasikan memuat 12 kolom (kolom `user_id` sengaja tidak
disertakan demi menjaga privasi pengguna):

| Kolom | Keterangan |
|-------|-----------|
| `review_id` | Identitas unik tiap ulasan |
| `nama_tempat` | Nama atraksi wisata yang diulas |
| `kabupaten` | Wilayah administratif tempat atraksi berada |
| `provinsi` | Provinsi (Bali) |
| `rating` | Rating pengguna, skala 1–5 |
| `review` | Teks ulasan asli |
| `review_clean` | Teks ulasan setelah prapemrosesan ringan |
| `language` | Bahasa terdeteksi (`en` / `id` / `es`) |
| `proxy_sentiment` | Label acuan hasil pemetaan rating (*distant supervision*) |
| `sentiment_label` | **Label sentimen akhir hasil model terpilih** |
| `confidence_score` | Skor keyakinan model terhadap `sentiment_label` |
| `sumber` | Platform asal ulasan (TripAdvisor / Google Review) |

Tiga kolom label penting untuk dipahami perbedaannya:

- **`proxy_sentiment`** — label acuan yang dibentuk otomatis dari rating
  (lihat skema di bawah). Diperlakukan sebagai **distant supervision label**,
  bukan *gold standard* maupun hasil anotasi manual manusia.
- **`sentiment_label`** — label sentimen final pada dataset, hasil pelabelan
  otomatis oleh model terpilih (XLM-RoBERTa fine-tuned) atas seluruh baris.
- **`confidence_score`** — tingkat keyakinan model saat memberikan
  `sentiment_label` pada baris tersebut.

### Skema Proxy Label

Label acuan (`proxy_sentiment`) dibentuk melalui pemetaan rating pengguna:

| Rating | Proxy Sentiment |
|--------|-----------------|
| 1 – 2  | negatif |
| 3      | netral  |
| 4 – 5  | positif |

---

## Model yang Dievaluasi

Lima pendekatan dievaluasi pada **test set** yang sama (1.617 baris), dengan
**macro F1-score** sebagai metrik utama. Angka diurutkan dari macro F1
tertinggi:

| Model | Pendekatan | Accuracy | Macro F1 | Weighted F1 |
|-------|------------|:--------:|:--------:|:-----------:|
| GPT prompt-based zero-shot | Zero-shot (prompt-based) | 83,67% | 70,88% | 86,55% |
| **XLM-RoBERTa fine-tuned** ✅ | Fine-tuning | **89,98%** | **70,74%** | **89,68%** |
| mBERT fine-tuned | Fine-tuning | 88,13% | 63,82% | 87,14% |
| XLM-RoBERTa zero-shot | Zero-shot | 87,94% | 58,85% | 86,12% |
| mBERT-XNLI zero-shot | Zero-shot (NLI) | 79,47% | 46,97% | 78,62% |

> ✅ **Model terpilih untuk pelabelan final: XLM-RoBERTa fine-tuned.**

### Pemilihan Model Pelabelan

Jika diurutkan hanya berdasarkan macro F1, GPT prompt-based zero-shot berada
di peringkat teratas (70,88% vs 70,74%). Namun pemilihan model pelabelan pada
penelitian ini tidak didasarkan pada satu metrik, melainkan **tiga kriteria**:
macro F1, konsistensi antarkelas, dan kelayakan penerapan praktis. Berdasarkan
ketiganya, model yang dipilih adalah **XLM-RoBERTa fine-tuned**, dengan alasan:

1. **Macro F1 praktis setara.** Selisih dengan GPT hanya 0,14 poin
   (70,88% vs 70,74%) — terlalu tipis untuk dijadikan dasar tunggal.
2. **Konsistensi antarkelas lebih baik.** GPT memiliki jurang lebar antara
   macro precision (68,36%) dan macro recall (79,39%), sedangkan XLM-RoBERTa
   fine-tuned jauh lebih seimbang (72,29% dan 69,37%). Pada distribusi data
   sebenarnya, XLM-RoBERTa fine-tuned juga unggul: accuracy 89,98% vs 83,67%
   dan weighted F1 89,68% vs 86,55%.
3. **Kelayakan penerapan praktis.** XLM-RoBERTa fine-tuned adalah model
   *open-source* yang dijalankan lokal — sekali dilatih dapat melabeli seluruh
   dataset secara reproducible dan tanpa biaya per pemanggilan, sementara GPT
   prompt-based bergantung pada API eksternal berbayar.

Dengan model terpilih, seluruh 10.777 baris dilabeli untuk mengisi kolom
`sentiment_label`. Tingkat kesesuaian antara `proxy_sentiment` dan label model
pada keseluruhan dataset mencapai **93,28%**.

> **Catatan:** keluaran otomatis pada Notebook 03 menandai "MODEL TERBAIK"
> berdasarkan pengurutan macro F1 semata, sehingga menampilkan GPT. Penetapan
> model pelabelan final tetap XLM-RoBERTa fine-tuned sesuai tiga kriteria di
> atas.

---

## Struktur Repositori

```
nlp-labeling-attraction-reviews/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # Raw Data hasil pengumpulan (tanpa user_id)
│   ├── interim/        # Data antara: hasil cleaning, proxy label, split
│   └── final/          # Dataset final berlabel (10.777 baris, 12 kolom)
├── notebooks/          # Tiga notebook penelitian (01, 02, 03)
├── src/                # Fungsi inti tiap tahap penelitian
│   ├── cleaning.py     # Tahap 1: pembersihan data & verifikasi bahasa
│   ├── preprocessing.py# Tahap 2: prapemrosesan, proxy label, split dataset
│   └── modeling.py     # Tahap 3: evaluasi lima model & pelabelan dataset
└── results/            # Metrik evaluasi & artefak hasil (mis. confusion matrix)
```

> Catatan: berkas pada `src/` merupakan ekstraksi fungsi inti dari tiga
> notebook penelitian. Sel eksplorasi, pencetakan progres, dan visualisasi
> tidak disertakan agar kode lebih ringkas dan mudah dibaca. Notebook versi
> lengkap tersedia pada folder `notebooks/`.

---

## Alur Penelitian

Proses data disusun mengikuti empat tahap berurutan, dari data mentah hingga
dataset final berlabel:

**1. Pengumpulan Data.** Ulasan dikumpulkan melalui *scraping* dari dua
platform: TripAdvisor (6.768 ulasan) dan Google Review (15.360 ulasan),
menghasilkan 22.128 baris mentah beserta atribut pendukung seperti rating,
nama tempat, dan lokasi.

**2. Pembersihan & Verifikasi Bahasa** — `cleaning.py` (Notebook 01).
Standardisasi kolom, pembersihan bertahap (menghapus ulasan kosong, duplikat,
dan ulasan terlalu pendek), penyaringan bahasa berdasarkan label platform,
verifikasi bahasa menggunakan `langdetect`, serta penyaringan akhir hingga
tersisa tiga bahasa target (Inggris, Indonesia, Spanyol).

**3. Prapemrosesan, Proxy Label & Split** — `preprocessing.py` (Notebook 02).
Teks melewati prapemrosesan ringan — **tanpa** *stopword removal*, *stemming*,
maupun *lemmatization* — agar konteks sentimen tetap utuh. Label acuan dibentuk
melalui *distant supervision*: rating dipetakan menjadi `proxy_sentiment`
(1–2 = negatif, 3 = netral, 4–5 = positif). Setelah tahap ini dataset berjumlah
10.777 baris, lalu dibagi secara *stratified* 70:15:15
(train 7.543 / validation 1.617 / test 1.617; `random_state = 42`,
distratifikasi atas kombinasi bahasa × sentimen).

**4. Evaluasi & Pelabelan Final** — `modeling.py` (Notebook 03). Lima
pendekatan model dievaluasi pada test set yang sama (lihat tabel di atas).
Model terbaik menurut tiga kriteria, XLM-RoBERTa fine-tuned, kemudian digunakan
untuk melabeli seluruh 10.777 baris dan mengisi kolom `sentiment_label` beserta
`confidence_score`. Hasil akhirnya adalah dataset berlabel sentimen yang
menjadi output utama penelitian ini.

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
- `proxy_sentiment` adalah label acuan (*distant supervision*), bukan hasil
  anotasi manusia; `sentiment_label` adalah label akhir hasil model.
- Fine-tuning dijalankan pada lingkungan ber-GPU (Google Colab).
