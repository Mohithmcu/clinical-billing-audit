#!/usr/bin/env python3
"""Generate the full suite of production-grade Jupyter notebooks for the project."""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_all_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────
    # 01_data_exploration.ipynb
    # ────────────────────────────────────────────────────────────
    nb01 = make_notebook([
        md_cell("""# 01. Clinical Notes & Claims Data Exploration
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

In this notebook, we explore the primary dataset (MTSamples clinical transcriptions) and synthetic CMS DE-SynPUF claims, perform statistical profiling, and execute automated data-quality checks."""),
        code_cell("""import sys
from pathlib import Path

# Add project root src to path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clinical_billing_audit.data.loaders import load_mtsamples, load_cms_claims, load_icd9_mapping
from clinical_billing_audit.data.cleaners import clean_clinical_notes, clean_specialty_labels, profile_dataset
from clinical_billing_audit.data.quality import DataQualityChecker

sns.set_theme(style="whitegrid")
%matplotlib inline"""),
        md_cell("## 1. Load MTSamples Clinical Notes"),
        code_cell("""notes_raw = load_mtsamples("../data/raw/mtsamples.csv")
print(f"Total notes loaded: {len(notes_raw):,}")
notes_raw.head(3)"""),
        md_cell("## 2. Clean Notes & Specialties"),
        code_cell("""notes_clean = clean_clinical_notes(notes_raw, min_length=50)
notes_clean = clean_specialty_labels(notes_clean)
print(f"Cleaned notes count: {len(notes_clean):,}")
print(f"Unique medical specialties: {notes_clean['medical_specialty'].nunique()}")"""),
        md_cell("## 3. Data Quality Audit & Profiling"),
        code_cell("""qc = DataQualityChecker()
report = qc.run(notes_clean, dataset_type="notes")
report_df = report.summary_df()
print("Data Quality Summary:")
report_df"""),
        md_cell("## 4. Specialty Distribution & Note Lengths"),
        code_cell("""notes_clean['char_length'] = notes_clean['transcription'].str.len()
notes_clean['word_count'] = notes_clean['transcription'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 Specialties
top_specs = notes_clean['medical_specialty'].value_counts().head(15)
sns.barplot(x=top_specs.values, y=top_specs.index, ax=axes[0], palette="Blues_r")
axes[0].set_title("Top 15 Medical Specialties in Dataset", fontsize=13)
axes[0].set_xlabel("Number of Notes")

# Note Length Distribution
sns.histplot(notes_clean['word_count'], bins=40, kde=True, ax=axes[1], color="#2563eb")
axes[1].set_title("Note Word Count Distribution", fontsize=13)
axes[1].set_xlabel("Word Count")

plt.tight_layout()
plt.show()"""),
        md_cell("## 5. Summary & Key Findings\n- Notes represent diverse clinical encounter types.\n- Word counts range from brief consultations to detailed operative reports.\n- Data is ready for NLP entity extraction."),
    ])

    # ────────────────────────────────────────────────────────────
    # 02_ner_analysis.ipynb
    # ────────────────────────────────────────────────────────────
    nb02 = make_notebook([
        md_cell("""# 02. Biomedical Named Entity Recognition (NER) & Code Mapping
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

Here we extract diagnoses (DISEASE), medications (CHEMICAL), procedures (PROCEDURE), and anatomical sites (ANATOMY) from clinical notes and map them to standardized ICD-10 / HCPCS code categories."""),
        code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clinical_billing_audit.nlp.ner_extractor import NERExtractor
from clinical_billing_audit.nlp.entity_normalizer import normalize_entity
from clinical_billing_audit.nlp.code_mapper import entity_to_code_category, predict_code_categories_from_entities"""),
        md_cell("## 1. Extract Entities from Sample Note"),
        code_cell("""extractor = NERExtractor()

sample_note = (
    "HISTORY: 64-year-old male with severe essential hypertension and chronic congestive heart failure. "
    "Presents with worsening dyspnea. Underwent diagnostic colonoscopy showing diverticulosis. "
    "Current medications include furosemide 40 mg daily, lisinopril 20 mg, and aspirin."
)

result = extractor.extract(sample_note, note_id="DEMO_01")
print(f"Extracted {len(result.entities)} entities:")
print(f"- Diagnoses:   {result.diagnoses}")
print(f"- Medications: {result.medications}")
print(f"- Procedures:  {result.procedures}")
print(f"- Anatomy:     {result.anatomy}")"""),
        md_cell("## 2. Entity Normalization & Code Category Mapping"),
        code_cell("""normalized_diags = [normalize_entity(d, "DISEASE") for d in result.diagnoses]
normalized_procs = [normalize_entity(p, "PROCEDURE") for p in result.procedures]

print("Normalized Diagnoses:", normalized_diags)
print("Normalized Procedures:", normalized_procs)

category_votes = predict_code_categories_from_entities(normalized_diags, normalized_procs)
print("\\nPredicted Code Categories (from entities):")
for cat, score in category_votes.items():
    print(f"  {cat:<25}: {score:.1%}")"""),
    ])

    # ────────────────────────────────────────────────────────────
    # 03_model_training.ipynb
    # ────────────────────────────────────────────────────────────
    nb03 = make_notebook([
        md_cell("""# 03. Baseline Classifier Training (TF-IDF + Logistic Regression)
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

Trains a CPU-friendly, interpretable multi-class baseline classifier predicting the expected code category from note content, combined with text-length and entity count features."""),
        code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from clinical_billing_audit.models.baseline import BaselineClassifier"""),
        md_cell("## 1. Load Processed Paired Dataset"),
        code_cell("""data_path = Path("../data/processed/paired_with_mismatches.csv")
if not data_path.exists():
    print("Paired dataset not found. Run pipeline first or load from raw.")
    df = pd.read_csv("../data/raw/mtsamples.csv").dropna(subset=['transcription'])
    df['code_category'] = 'general'
else:
    df = pd.read_csv(data_path)

print(f"Total dataset size: {len(df):,}")"""),
        md_cell("## 2. Stratified Train / Test Split"),
        code_cell("""texts = df['transcription'].tolist()
labels = df['code_category'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")"""),
        md_cell("## 3. Train Baseline Model & Cross-Validation"),
        code_cell("""clf = BaselineClassifier(
    tfidf_params={"max_features": 15000, "ngram_range": (1, 2)},
    lr_params={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
)

train_res = clf.fit(X_train, y_train)
print(f"Training Weighted F1: {train_res['train_f1_weighted']:.4f}")

eval_report = clf.evaluate(X_test, y_test)
print(f"Test Accuracy: {eval_report['accuracy']:.4f}")
print(f"Test Weighted F1: {eval_report['weighted avg']['f1-score']:.4f}")"""),
        md_cell("## 4. Top Feature Importances per Category"),
        code_cell("""importance_df = clf.get_feature_importance(top_n=10)
importance_df.head(15)"""),
    ])

    # ────────────────────────────────────────────────────────────
    # 03b_transformer_training.ipynb (KAGGLE GPU)
    # ────────────────────────────────────────────────────────────
    nb03b = make_notebook([
        md_cell("""# 03b. Transformer Fine-Tuning on Kaggle (BioBERT / ClinicalBERT)
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

> **Execution Environment**: Designed for Kaggle Notebooks with **GPU accelerator enabled** (NVIDIA Tesla P100 or T4).
>
> **Goal**: Fine-tune `dmis-lab/biobert-base-cased-v1.2` or `emilyalsentzer/Bio_ClinicalBERT` on clinical transcriptions to predict ICD/HCPCS code categories with high fidelity (~88–92% F1)."""),
        code_cell("""# 1. Environment & GPU Verification
import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"CUDA Available: {gpu_available}")
if gpu_available:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: GPU not detected. Enable GPU in Kaggle settings: Settings -> Accelerator -> GPU P100.")"""),
        md_cell("## 2. Install & Import Hugging Face Libraries"),
        code_cell("""!pip install -q transformers datasets accelerate evaluate scikit-learn

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate"""),
        md_cell("## 3. Load & Preprocess MTSamples Dataset"),
        code_cell("""# Kaggle path to dataset
KAGGLE_PATH = "/kaggle/input/medicaltranscriptions/mtsamples.csv"
LOCAL_PATH = "data/raw/mtsamples.csv"

data_file = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_PATH
df = pd.read_csv(data_file)
df = df.dropna(subset=['transcription', 'medical_specialty']).reset_index(drop=True)

# Map specialty to code category
SPECIALTY_MAP = {
    "Cardiovascular / Pulmonary": "circulatory",
    "Orthopedic": "musculoskeletal",
    "Gastroenterology": "digestive",
    "Neurology": "nervous",
    "Surgery": "surgical",
    "Urology": "genitourinary",
    "Dermatology": "skin_subcutaneous",
    "General Medicine": "general",
}
df['category'] = df['medical_specialty'].map(lambda s: SPECIALTY_MAP.get(s, "general"))

# Label Encoding
labels = sorted(df['category'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}
df['label'] = df['category'].map(label2id)

print(f"Total samples: {len(df):,}")
print(f"Classes ({len(labels)}): {labels}")"""),
        md_cell("## 4. Tokenization & Dataset Preparation"),
        code_cell("""MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Train/Val/Test Split (80/10/10)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

def tokenize_function(examples):
    return tokenizer(examples['transcription'], truncation=True, max_length=512)

train_ds = Dataset.from_pandas(train_df[['transcription', 'label']]).map(tokenize_function, batched=True)
val_ds = Dataset.from_pandas(val_df[['transcription', 'label']]).map(tokenize_function, batched=True)
test_ds = Dataset.from_pandas(test_df[['transcription', 'label']]).map(tokenize_function, batched=True)"""),
        md_cell("## 5. Model Initialization & Trainer Setup"),
        code_cell("""model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_weighted": f1_weighted, "f1_macro": f1_macro}

training_args = TrainingArguments(
    output_dir="./biobert_clinical_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)"""),
        md_cell("## 6. Fine-Tuning & Evaluation"),
        code_cell("""# Run Training
trainer.train()

# Evaluate on held-out test set
test_results = trainer.evaluate(test_ds)
print("=== HELD-OUT TEST SET PERFORMANCE ===")
for k, v in test_results.items():
    print(f"  {k}: {v:.4f}")"""),
        md_cell("## 7. Export Model Weights & Config for Local Inference"),
        code_cell("""# Save trained model to Kaggle output
output_export_dir = "./exported_biobert_model"
model.save_pretrained(output_export_dir)
tokenizer.save_pretrained(output_export_dir)

print(f"✅ Model successfully saved to {output_export_dir}")
print("Download these artifacts from Kaggle's Data Output tab to models/transformer/ locally.")"""),
    ])

    # ────────────────────────────────────────────────────────────
    # 04_anomaly_detection.ipynb
    # ────────────────────────────────────────────────────────────
    nb04 = make_notebook([
        md_cell("""# 04. Anomaly Detection & Mismatch Scoring (Isolation Forest)
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

In this notebook, we engineer clinical and coding deviation features and train an Isolation Forest to score clinical note-claim pairs for potential billing mismatches."""),
        code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clinical_billing_audit.models.anomaly import AnomalyDetector
from clinical_billing_audit.models.ensemble import EnsembleMismatchScorer"""),
        md_cell("## 1. Load Paired Results & Engineer Features"),
        code_cell("""df = pd.read_csv("../data/processed/paired_with_mismatches.csv")
detector = AnomalyDetector(n_estimators=200, contamination=0.15)
features = detector.engineer_features(df)
print("Engineered Features Shape:", features.shape)
features.head()"""),
        md_cell("## 2. Fit Isolation Forest & Compute Anomaly Scores"),
        code_cell("""detector.fit(features)
scores = detector.score(features)
flagged_df = detector.flag_mismatches(df, features)

plt.figure(figsize=(10, 5))
sns.histplot(data=flagged_df, x="anomaly_score", hue="is_mismatch", bins=30, kde=True, palette="coolwarm")
plt.title("Isolation Forest Anomaly Score Distribution (Clean vs Injected Mismatch)", fontsize=13)
plt.xlabel("Anomaly Score (Higher = More Suspicious)")
plt.show()"""),
    ])

    # ────────────────────────────────────────────────────────────
    # 05_evaluation.ipynb
    # ────────────────────────────────────────────────────────────
    nb05 = make_notebook([
        md_cell("""# 05. Model Evaluation, PR Curves & Human-Review Benchmarks
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

Calculates rigorous classification metrics (Precision, Recall, F1, PR-AUC, ROC-AUC) across threshold sweeps and benchmarks performance against human audit simulations."""),
        code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clinical_billing_audit.evaluation.metrics import (
    compute_binary_metrics,
    compute_threshold_sweep,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    generate_evaluation_report
)"""),
        md_cell("## 1. Load Flagged Pipeline Results"),
        code_cell("""results_df = pd.read_csv("../data/processed/flagged_results.csv")
y_true = results_df['is_mismatch'].astype(int).values
y_pred = results_df['ensemble_flag'].astype(int).values
y_score = results_df['ensemble_score'].values

metrics = compute_binary_metrics(y_true, y_pred, y_score)
for k, v in metrics.items():
    print(f"{k:<18}: {v:.4f}")"""),
        md_cell("## 2. Threshold Sweep Analysis"),
        code_cell("""threshold_df = compute_threshold_sweep(y_true, y_score)
plt.figure(figsize=(9, 5))
plt.plot(threshold_df['threshold'], threshold_df['precision'], label="Precision", color="#2563eb", lw=2)
plt.plot(threshold_df['threshold'], threshold_df['recall'], label="Recall", color="#dc2626", lw=2)
plt.plot(threshold_df['threshold'], threshold_df['f1'], label="F1 Score", color="#059669", lw=2)
plt.xlabel("Ensemble Score Threshold")
plt.ylabel("Metric Value")
plt.title("Precision, Recall, and F1 across Decision Thresholds", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),
    ])

    # ────────────────────────────────────────────────────────────
    # 06_error_analysis.ipynb
    # ────────────────────────────────────────────────────────────
    nb06 = make_notebook([
        md_cell("""# 06. Error Analysis & False Positive Triage
### Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

Performs stratified failure-mode analysis on false positives and false negatives to diagnose pipeline bottlenecks across clinical specialties and mismatch types."""),
        code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clinical_billing_audit.evaluation.error_analysis import (
    error_breakdown,
    error_by_mismatch_type,
    error_by_note_length,
    top_false_positives,
    top_false_negatives,
    plot_error_analysis
)"""),
        md_cell("## 1. Error Breakdown by Medical Specialty"),
        code_cell("""results_df = pd.read_csv("../data/processed/flagged_results.csv")
spec_df = error_breakdown(results_df, y_pred_col="ensemble_flag", group_col="medical_specialty")
spec_df.head(10)"""),
        md_cell("## 2. Error Breakdown by Mismatch Type"),
        code_cell("""mtype_df = error_by_mismatch_type(results_df, y_pred_col="ensemble_flag")
mtype_df"""),
        md_cell("## 3. Top False Positives (Audit Triage)"),
        code_cell("""fps = top_false_positives(results_df, score_col="ensemble_score", y_pred_col="ensemble_flag", n=5)
fps[['pair_id', 'medical_specialty', 'ensemble_score', 'transcription']]"""),
    ])

    # Write all notebooks
    notebooks = {
        "01_data_exploration.ipynb": nb01,
        "02_ner_analysis.ipynb": nb02,
        "03_model_training.ipynb": nb03,
        "03b_transformer_training.ipynb": nb03b,
        "04_anomaly_detection.ipynb": nb04,
        "05_evaluation.ipynb": nb05,
        "06_error_analysis.ipynb": nb06,
    }

    for name, content in notebooks.items():
        filepath = nb_dir / name
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        print(f"Created notebook: {filepath}")


if __name__ == "__main__":
    build_all_notebooks()
