"""
preprocessing.py
================
Fungsi-fungsi inti untuk tahap prapemrosesan ringan, pembentukan proxy
label, dan pembagian dataset (Notebook 02: Preprocessing, Proxy Label,
& Split Dataset).

Karakteristik tahap ini:
    - Prapemrosesan bersifat RINGAN: hanya HTML decode, penghapusan tag,
      URL, email, karakter kontrol, dan perapian spasi. TIDAK dilakukan
      stopword removal, stemming, maupun lemmatization, karena dataset
      ditujukan untuk model berbasis Transformer dan LLM.
    - Proxy label sentimen dibentuk dari pemetaan rating dan diperlakukan
      sebagai silver label (distant supervision), bukan gold standard.
    - Pembagian dataset 70:15:15 menggunakan stratified sampling
      berdasarkan kombinasi bahasa x proxy_sentiment.

Catatan:
    Kode ini merupakan ekstraksi fungsi inti dari notebook. Sel eksplorasi,
    pencetakan progres, dan visualisasi tidak disertakan. Untuk reproduksi
    penuh beserta luarannya, lihat notebook pada folder notebooks/.
"""

import re
import html

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

SEED = 42

# Kolom wajib untuk validasi dataset
REQUIRED_COLS = ["review", "rating", "language"]

# Urutan kolom dataset final (13 kolom)
KOLOM_FINAL = [
    "review_id", "user_id", "nama_tempat", "id_tempat", "rating",
    "kabupaten", "provinsi", "review", "language", "language_original",
    "proxy_sentiment", "sumber", "review_clean",
]


def validate_dataframe(df):
    """Memvalidasi kolom wajib dan membersihkan nilai yang tidak valid.

    Tahapan:
        - Memastikan kolom wajib tersedia.
        - Menghapus baris dengan review kosong.
        - Memastikan rating numerik dan berada pada rentang 1-5.

    Args:
        df (pd.DataFrame): dataframe bersih hasil Notebook 01.

    Returns:
        pd.DataFrame: dataframe yang telah tervalidasi.

    Raises:
        ValueError: jika ada kolom wajib yang tidak ditemukan.
    """
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["review"] = df["review"].astype(str)
    df["review"] = df["review"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["review"]).copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)
    df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()

    return df.reset_index(drop=True)


def light_text_preprocessing(text):
    """Prapemrosesan ringan untuk model Transformer dan LLM.

    TIDAK melakukan stopword removal, stemming, atau lemmatization.
    Langkah yang dilakukan: decode HTML entity, hapus HTML tag, URL,
    email, karakter kontrol, lalu rapikan spasi.

    Args:
        text (str): teks ulasan mentah.

    Returns:
        str: teks ulasan yang telah diproses.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)              # 1. decode HTML entity (&amp; -> &)
    text = re.sub(r"<.*?>", " ", text)      # 2. hapus HTML tag
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # 3. hapus URL
    text = re.sub(r"\S+@\S+", " ", text)    # 4. hapus email
    text = re.sub(r"[\r\n\t]+", " ", text)  # 5. hapus karakter kontrol
    text = re.sub(r"\s+", " ", text)        # 6. rapikan spasi berlebih
    text = text.strip()                     # 7. strip kiri-kanan
    return text


def rating_to_sentiment(rating):
    """Memetakan nilai rating (1-5) ke label sentimen proxy (silver label).

    Skema pemetaan:
        rating 1-2 -> negatif
        rating 3   -> netral
        rating 4-5 -> positif

    Args:
        rating (int): nilai rating ulasan.

    Returns:
        str: label proxy sentiment, atau np.nan jika di luar rentang.
    """
    if rating in [1, 2]:
        return "negatif"
    elif rating == 3:
        return "netral"
    elif rating in [4, 5]:
        return "positif"
    else:
        return np.nan


def compute_imbalance_ratio(sentiment_counts):
    """Menghitung Imbalance Ratio (IR) dari distribusi kelas.

    IR = jumlah kelas mayoritas / jumlah kelas minoritas.

    Args:
        sentiment_counts (pd.Series): hasil value_counts() proxy_sentiment.

    Returns:
        float: nilai Imbalance Ratio.
    """
    return sentiment_counts.max() / sentiment_counts.min()


def apply_preprocessing(df):
    """Menerapkan prapemrosesan ringan dan pembentukan proxy label.

    Menambahkan kolom 'review_clean' dan 'proxy_sentiment', lalu menghapus
    baris yang teksnya kosong setelah prapemrosesan.

    Args:
        df (pd.DataFrame): dataframe tervalidasi.

    Returns:
        pd.DataFrame: dataframe dengan kolom hasil prapemrosesan dan proxy label.
    """
    df["review_clean"] = df["review"].apply(light_text_preprocessing)
    df = df[df["review_clean"].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    df["proxy_sentiment"] = df["rating"].apply(rating_to_sentiment)
    assert df["proxy_sentiment"].isnull().sum() == 0, \
        "Ada rating yang gagal dipetakan ke proxy_sentiment."

    return df


def finalize_columns(df):
    """Menghapus duplikat akhir, menambahkan review_id, dan menyusun kolom final.

    Args:
        df (pd.DataFrame): dataframe hasil apply_preprocessing.

    Returns:
        pd.DataFrame: dataframe final (13 kolom) siap dibagi.
    """
    # Hapus duplikat berdasarkan hasil prapemrosesan
    if "nama_tempat" in df.columns:
        df = df.drop_duplicates(subset=["nama_tempat", "review_clean", "rating"]).copy()
    else:
        df = df.drop_duplicates(subset=["review_clean", "rating"]).copy()
    df = df.reset_index(drop=True)

    # Tambahkan review_id
    if "review_id" in df.columns:
        df = df.drop(columns=["review_id"])
    df.insert(0, "review_id", range(1, len(df) + 1))

    # Susun urutan kolom final
    kolom_final_ada = [c for c in KOLOM_FINAL if c in df.columns]
    return df[kolom_final_ada].copy()


def stratified_split(df, seed=SEED):
    """Membagi dataset menjadi train/validation/test dengan proporsi 70:15:15.

    Stratifikasi dilakukan berdasarkan kombinasi bahasa x proxy_sentiment
    agar proporsi setiap kelas terjaga di seluruh subset.

    Args:
        df (pd.DataFrame): dataframe final.
        seed (int): random_state untuk reproduktibilitas.

    Returns:
        tuple: (train_df, val_df, test_df), masing-masing pd.DataFrame.
    """
    # Kolom stratifikasi gabungan
    df = df.copy()
    df["stratify_col"] = df["language"] + "_" + df["proxy_sentiment"]

    # Tahap 1: pisahkan train (70%) dari sisanya (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["stratify_col"]
    )

    # Tahap 2: bagi sisanya menjadi validation (15%) dan test (15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["stratify_col"]
    )

    # Bersihkan kolom bantu
    train_df = train_df.drop(columns=["stratify_col"]).reset_index(drop=True)
    val_df   = val_df.drop(columns=["stratify_col"]).reset_index(drop=True)
    test_df  = test_df.drop(columns=["stratify_col"]).reset_index(drop=True)

    return train_df, val_df, test_df


def compare_proportions(df, train_df, val_df, test_df, col):
    """Menyusun tabel perbandingan proporsi (%) suatu kolom antar subset.

    Berguna untuk memverifikasi bahwa stratified sampling menjaga
    representasi kelas secara konsisten.

    Args:
        df (pd.DataFrame): dataset penuh.
        train_df, val_df, test_df (pd.DataFrame): subset hasil pembagian.
        col (str): nama kolom yang diperiksa (mis. 'proxy_sentiment').

    Returns:
        pd.DataFrame: tabel proporsi (%) per subset.
    """
    return pd.DataFrame({
        "Dataset Penuh": df[col].value_counts(normalize=True).mul(100).round(2),
        "Train"        : train_df[col].value_counts(normalize=True).mul(100).round(2),
        "Validation"   : val_df[col].value_counts(normalize=True).mul(100).round(2),
        "Test"         : test_df[col].value_counts(normalize=True).mul(100).round(2),
    })
