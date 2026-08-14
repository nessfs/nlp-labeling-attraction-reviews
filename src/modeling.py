"""
modeling.py
===========
Fungsi-fungsi inti untuk implementasi, evaluasi komparatif lima pendekatan
model NLP, dan pelabelan dataset penuh (Notebook 03: Implementasi, Evaluasi,
& Pelabelan).

Lima pendekatan yang dievaluasi pada test set yang sama:
    1. GPT prompt-based zero-shot   (prediksi dibaca dari kolom gpt_pred)
    2. mBERT-XNLI zero-shot         (NLI-based zero-shot)
    3. XLM-RoBERTa zero-shot        (NLI-based zero-shot)
    4. mBERT fine-tuned             (fine-tuning)
    5. XLM-RoBERTa fine-tuned       (fine-tuning) -> model terbaik

Metrik utama: macro F1-score (label acuan = proxy_sentiment).

Catatan implementasi:
    - Pustaka `datasets` dari Hugging Face sengaja TIDAK dipakai untuk
      fine-tuning karena pemanggilan internalnya ke torchvision.io dapat
      menimbulkan ImportError pada versi torchvision terbaru. Sebagai
      gantinya digunakan kelas SentimentDataset berbasis PyTorch.
    - make_training_args() menyediakan fallback nama parameter strategi
      evaluasi ('eval_strategy' vs 'evaluation_strategy') agar kompatibel
      lintas versi transformers.

    Kode ini merupakan ekstraksi fungsi inti dari notebook. Sel eksplorasi,
    pencetakan progres, dan visualisasi tidak disertakan. Untuk reproduksi
    penuh beserta luarannya, lihat notebook pada folder notebooks/.
"""

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding,
)

# ── Konfigurasi global ───────────────────────────────────────────────────────
SEED = 42

LABEL_ORDER = ["negatif", "netral", "positif"]
label2id = {"negatif": 0, "netral": 1, "positif": 2}
id2label = {0: "negatif", 1: "netral", 2: "positif"}

# Hyperparameter fine-tuning
MAX_LENGTH       = 128
NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 32

# Konfigurasi zero-shot (NLI)
CANDIDATE_LABELS    = ["positive", "neutral", "negative"]
HYPOTHESIS_TEMPLATE = "This tourism review is {}."
LABEL_EN_TO_ID = {"positive": "positif", "neutral": "netral", "negative": "negatif"}

# Checkpoint model
MBERT_ZERO_SHOT_CKPT = "AyoubChLin/bert-base-multilingual-cased-xnli-nli"
XLMR_ZERO_SHOT_CKPT  = "joeddav/xlm-roberta-large-xnli"
MBERT_FINETUNE_CKPT  = "bert-base-multilingual-cased"
XLMR_FINETUNE_CKPT   = "xlm-roberta-base"

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_label(label):
    """Menormalisasi variasi penulisan label ke tiga kelas standar.

    Menangani perbedaan bahasa (positive/positif), huruf besar-kecil,
    serta tanda baca yang mungkin ikut terbawa dari keluaran model.

    Args:
        label (str): label mentah.

    Returns:
        str: 'negatif', 'netral', 'positif', atau 'unknown'.
    """
    label = str(label).strip().lower()
    for ch in [".", ",", ";", "`", '"', "'", "!", "?", ":"]:
        label = label.replace(ch, "")
    label = label.strip()

    if "positive" in label or "positif" in label:
        return "positif"
    elif "neutral" in label or "netral" in label:
        return "netral"
    elif "negative" in label or "negatif" in label:
        return "negatif"
    else:
        return "unknown"


def evaluate_predictions(y_true, y_pred, model_name):
    """Menghitung metrik evaluasi lengkap untuk satu model.

    Hanya baris dengan label acuan dan prediksi yang valid (termasuk
    salah satu dari LABEL_ORDER) yang diperhitungkan.

    Args:
        y_true (iterable): label acuan (proxy_sentiment).
        y_pred (iterable): label prediksi model.
        model_name (str): nama model untuk pelabelan hasil.

    Returns:
        tuple: (result (dict), report (dict), cm (np.ndarray))
            - result : ringkasan metrik utama (accuracy, macro/weighted P/R/F1)
            - report : classification report per kelas (dict)
            - cm     : confusion matrix (urutan LABEL_ORDER)
    """
    y_true = [normalize_label(x) for x in y_true]
    y_pred = [normalize_label(x) for x in y_pred]

    valid_idx = [
        i for i, pred in enumerate(y_pred)
        if pred in LABEL_ORDER and y_true[i] in LABEL_ORDER
    ]
    y_true_valid = [y_true[i] for i in valid_idx]
    y_pred_valid = [y_pred[i] for i in valid_idx]

    acc = accuracy_score(y_true_valid, y_pred_valid)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid,
        labels=LABEL_ORDER, average="macro", zero_division=0,
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid,
        labels=LABEL_ORDER, average="weighted", zero_division=0,
    )
    report = classification_report(
        y_true_valid, y_pred_valid,
        labels=LABEL_ORDER, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=LABEL_ORDER)

    result = {
        "model"             : model_name,
        "n_data"            : len(y_true_valid),
        "accuracy"          : acc,
        "macro_precision"   : p_macro,
        "macro_recall"      : r_macro,
        "macro_f1"          : f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall"   : r_weighted,
        "weighted_f1"       : f1_weighted,
    }
    return result, report, cm


def format_report(report):
    """Mengubah classification report (dict) menjadi DataFrame persen yang rapi.

    Args:
        report (dict): keluaran classification_report(output_dict=True).

    Returns:
        pd.DataFrame: tabel Precision/Recall/F1/Support per kelas (dalam %).
    """
    rows = []
    for kelas in LABEL_ORDER:
        rows.append({
            "Kelas"    : kelas,
            "Precision": round(report[kelas]["precision"] * 100, 2),
            "Recall"   : round(report[kelas]["recall"] * 100, 2),
            "F1-score" : round(report[kelas]["f1-score"] * 100, 2),
            "Support"  : int(report[kelas]["support"]),
        })
    return pd.DataFrame(rows)


class SentimentDataset(TorchDataset):
    """Dataset PyTorch untuk klasifikasi sentimen berbasis Transformer.

    Digunakan sebagai pengganti datasets.Dataset dari Hugging Face untuk
    menghindari ketergantungan pada torchvision.io yang dapat menimbulkan
    ImportError pada versi torchvision terbaru.

    Args:
        texts (list): daftar teks ulasan.
        labels (list): daftar label numerik (0=negatif, 1=netral, 2=positif).
        tokenizer: tokenizer dari model yang digunakan.
        max_length (int): panjang maksimum token.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def make_training_args(output_dir):
    """Membuat TrainingArguments dengan konfigurasi seragam.

    Menyediakan fallback nama parameter strategi evaluasi agar kompatibel
    antara transformers versi baru ('eval_strategy') dan lama
    ('evaluation_strategy').

    Args:
        output_dir (str): direktori luaran pelatihan.

    Returns:
        TrainingArguments: konfigurasi pelatihan.
    """
    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **base_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **base_kwargs)


def compute_metrics(eval_pred):
    """Menghitung metrik evaluasi pada setiap akhir epoch (untuk Trainer).

    Args:
        eval_pred (tuple): (logits, labels) dari Trainer.

    Returns:
        dict: accuracy, macro precision/recall/F1, dan weighted F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0,
    )
    return {
        "accuracy"       : acc,
        "macro_precision": p_macro,
        "macro_recall"   : r_macro,
        "macro_f1"       : f1_macro,
        "weighted_f1"    : f1_weighted,
    }


def run_zero_shot_model(test_df, text_col, checkpoint, model_label,
                        pred_col, score_col, batch_size=8):
    """Menjalankan inferensi zero-shot classification (NLI) pada test set.

    Prediksi dan skor disimpan ke test_df, lalu dievaluasi terhadap
    proxy_sentiment.

    Args:
        test_df (pd.DataFrame): test set (harus memuat text_col & proxy_sentiment).
        text_col (str): nama kolom teks input model.
        checkpoint (str): nama checkpoint model NLI.
        model_label (str): nama model untuk pelabelan hasil.
        pred_col (str): nama kolom penyimpan prediksi.
        score_col (str): nama kolom penyimpan confidence score.
        batch_size (int): ukuran batch inferensi.

    Returns:
        tuple: (result (dict), report (dict), cm (np.ndarray))
    """
    classifier = pipeline(
        "zero-shot-classification",
        model=checkpoint, tokenizer=checkpoint,
        device=0 if torch.cuda.is_available() else -1,
    )

    texts = test_df[text_col].astype(str).tolist()
    preds, scores = [], []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        try:
            batch_results = classifier(
                batch_texts,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                truncation=True,
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            for result in batch_results:
                preds.append(LABEL_EN_TO_ID[result["labels"][0]])
                scores.append(result["scores"][0])
        except Exception:
            # Jika satu batch gagal, tandai sebagai 'unknown' agar proses lanjut
            for _ in batch_texts:
                preds.append("unknown")
                scores.append(0.0)

    test_df[pred_col] = preds
    test_df[score_col] = scores

    return evaluate_predictions(
        test_df["proxy_sentiment"], test_df[pred_col], model_name=model_label
    )


def run_finetuned_model(train_df, val_df, test_df, text_col,
                        checkpoint, model_label, pred_col, output_dir):
    """Melakukan fine-tuning model pre-trained lalu mengevaluasi pada test set.

    Checkpoint terbaik dipilih berdasarkan macro F1 pada validation set.
    Jika model final sudah tersedia pada '{output_dir}_final', pelatihan
    dilewati dan model langsung dimuat untuk evaluasi (berguna saat sesi
    komputasi terputus).

    Args:
        train_df, val_df, test_df (pd.DataFrame): subset data (memuat kolom
            'label' numerik dan text_col).
        text_col (str): nama kolom teks input model.
        checkpoint (str): checkpoint dasar model.
        model_label (str): nama model untuk pelabelan hasil.
        pred_col (str): nama kolom penyimpan prediksi test set.
        output_dir (str): direktori luaran pelatihan.

    Returns:
        tuple: (result (dict), report (dict), cm (np.ndarray))
    """
    import os

    final_dir = output_dir + "_final"

    tokenizer = AutoTokenizer.from_pretrained(
        final_dir if os.path.exists(os.path.join(final_dir, "config.json"))
        else checkpoint
    )

    test_dataset = SentimentDataset(
        texts=test_df[text_col].astype(str).tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer, max_length=MAX_LENGTH,
    )

    # ── Jalur cepat: model final sudah ada, cukup dimuat untuk evaluasi ──────
    if os.path.exists(os.path.join(final_dir, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(final_dir)
        model.to(device)
        model.eval()

        trainer = Trainer(
            model=model, args=make_training_args(output_dir),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        test_output = trainer.predict(test_dataset)
        pred_ids = np.argmax(test_output.predictions, axis=1)
        test_df[pred_col] = [id2label[i] for i in pred_ids]

        return evaluate_predictions(
            test_df["proxy_sentiment"], test_df[pred_col], model_name=model_label
        )

    # ── Jalur pelatihan penuh ───────────────────────────────────────────────
    train_dataset = SentimentDataset(
        texts=train_df[text_col].astype(str).tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer, max_length=MAX_LENGTH,
    )
    val_dataset = SentimentDataset(
        texts=val_df[text_col].astype(str).tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer, max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3, id2label=id2label, label2id=label2id,
    )
    model.to(device)

    trainer = Trainer(
        model=model, args=make_training_args(output_dir),
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(test_output.predictions, axis=1)
    test_df[pred_col] = [id2label[i] for i in pred_ids]

    result, report, cm = evaluate_predictions(
        test_df["proxy_sentiment"], test_df[pred_col], model_name=model_label
    )

    # Simpan model final agar dapat dipakai ulang untuk pelabelan penuh
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    return result, report, cm


def evaluate_by_language(df, text_col, pred_col, model_name):
    """Mengevaluasi performa satu model secara terpisah pada setiap bahasa.

    Args:
        df (pd.DataFrame): data berisi kolom 'language', pred_col, proxy_sentiment.
        text_col (str): (tidak dipakai langsung; disediakan untuk konsistensi API).
        pred_col (str): nama kolom prediksi model.
        model_name (str): nama model untuk pelabelan hasil.

    Returns:
        pd.DataFrame: metrik per bahasa.
    """
    rows = []
    for lang in sorted(df["language"].dropna().unique()):
        subset = df[df["language"] == lang]
        if len(subset) == 0:
            continue
        result, _, _ = evaluate_predictions(
            subset["proxy_sentiment"], subset[pred_col], model_name=model_name
        )
        result["language"] = lang
        result["n_data"] = len(subset)
        rows.append(result)
    return pd.DataFrame(rows)


def predict_full_dataset(model_dir, texts, batch_size=64, max_length=MAX_LENGTH):
    """Menjalankan inferensi model fine-tuned pada seluruh teks dataset.

    Digunakan pada tahap pelabelan penuh (Bagian II Notebook 03) untuk
    memberi label sentimen pada keseluruhan dataset menggunakan model terbaik.

    Args:
        model_dir (str): direktori model fine-tuned final.
        texts (list): daftar teks yang akan dilabeli.
        batch_size (int): ukuran batch inferensi.
        max_length (int): panjang maksimum token.

    Returns:
        tuple: (preds (list), scores (list))
            - preds  : label prediksi
            - scores : confidence score (probabilitas tertinggi)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    preds, scores = [], []
    total = len(texts)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(device)

            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            batch_scores, batch_ids = torch.max(probs, dim=-1)

            preds.extend([id2label[int(i)] for i in batch_ids.cpu().numpy()])
            scores.extend([float(s) for s in batch_scores.cpu().numpy()])

    return preds, scores
