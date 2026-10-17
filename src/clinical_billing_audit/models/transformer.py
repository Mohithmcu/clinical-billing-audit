"""Transformer-based classifier for clinical notes.

This module provides a wrapper around HuggingFace Transformers for
fine-tuning BioBERT / ClinicalBERT on the specialty classification task.

For **training**, use the Kaggle notebook (requires GPU).
For **inference**, this module loads a fine-tuned model and runs on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clinical_billing_audit.utils.logging_config import get_logger

logger = get_logger(__name__)


class TransformerClassifier:
    """BioBERT / ClinicalBERT classifier for clinical notes.

    This class supports:
    - Loading a pre-trained or fine-tuned transformer model
    - CPU-based inference (for local use)
    - GPU-based training (for Kaggle)

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path to fine-tuned model directory.
    max_length : int
        Maximum token sequence length.
    num_labels : int
        Number of output classes.
    label_map : dict[int, str] | None
        Mapping from label index to label name.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 512,
        num_labels: int = 10,
        label_map: dict[int, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.label_map = label_map or {}
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def _load_model(self) -> None:
        """Lazy-load the transformer model and tokenizer."""
        if self._is_loaded:
            return

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )
            self._model.eval()
            self._is_loaded = True
            logger.info("Loaded transformer model: %s", self.model_name)

        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install 'clinical-billing-audit[transformers]'"
            )
        except Exception as e:
            logger.error("Failed to load model '%s': %s", self.model_name, e)
            raise

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict code categories for notes.

        Parameters
        ----------
        texts : list[str]
            Clinical note texts.

        Returns
        -------
        np.ndarray
            Predicted label indices.
        """
        import torch

        self._load_model()
        self._model.eval()

        predictions = []
        batch_size = 8  # Small batch for CPU

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self._tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**encodings)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).numpy()
                predictions.extend(preds.tolist())

        return np.array(predictions)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, num_labels)`` probability matrix.
        """
        import torch

        self._load_model()
        self._model.eval()

        all_probs = []
        batch_size = 8

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self._tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**encodings)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict_labels(self, texts: list[str]) -> list[str]:
        """Predict human-readable labels.

        Returns
        -------
        list[str]
            Predicted category names.
        """
        preds = self.predict(texts)
        return [self.label_map.get(int(p), f"class_{p}") for p in preds]

    def save_label_map(self, directory: str | Path) -> None:
        """Save the label mapping to JSON.

        Parameters
        ----------
        directory : str | Path
            Output directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "label_map.json", "w") as f:
            json.dump(self.label_map, f, indent=2)
        logger.info("Label map saved to %s", directory)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
    ) -> TransformerClassifier:
        """Load a fine-tuned model from disk.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing the fine-tuned model.

        Returns
        -------
        TransformerClassifier
            Loaded classifier.
        """
        model_dir = Path(model_dir)
        label_map = {}
        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = {int(k): v for k, v in json.load(f).items()}

        # Determine num_labels
        config_path = model_dir / "config.json"
        num_labels = 10
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                num_labels = config.get("num_labels", 10)

        model = cls(
            model_name=str(model_dir),
            num_labels=num_labels,
            label_map=label_map,
        )
        logger.info("Transformer classifier configured from %s", model_dir)
        return model


def generate_kaggle_training_script(
    output_path: str | Path,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    """Generate a Kaggle-ready training notebook as a Python script.

    Parameters
    ----------
    output_path : str | Path
        Where to write the script.
    model_name : str
        HuggingFace model to fine-tune.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Learning rate.
    """
    script = f'''#!/usr/bin/env python3
"""
Kaggle Training Script — Clinical Note Specialty Classifier
============================================================
Upload this script to Kaggle with GPU acceleration enabled.
Upload your processed dataset (processed_notes.csv) as a dataset.

Model: {model_name}
Task: Multi-class classification of clinical notes → code categories
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import torch

# ── Configuration ───────────────────────────────────────────
MODEL_NAME = "{model_name}"
MAX_LENGTH = 512
BATCH_SIZE = {batch_size}
NUM_EPOCHS = {num_epochs}
LEARNING_RATE = {learning_rate}
OUTPUT_DIR = "./fine_tuned_model"

# ── Load Data ───────────────────────────────────────────────
# Adjust path based on your Kaggle dataset upload
df = pd.read_csv("/kaggle/input/clinical-billing-data/processed_notes.csv")

# Use code_category as label
label_col = "code_category"
text_col = "transcription"

# Filter to categories with enough samples
min_samples = 20
category_counts = df[label_col].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
df = df[df[label_col].isin(valid_categories)].reset_index(drop=True)
print(f"Using {{len(valid_categories)}} categories, {{len(df)}} samples")

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df[label_col])
num_labels = len(le.classes_)
label_map = dict(enumerate(le.classes_))
print(f"Label map: {{label_map}}")

# Train/val split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
print(f"Train: {{len(train_df)}}, Val: {{len(val_df)}}")

# ── Tokenization ───────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_ds = Dataset.from_pandas(train_df[["transcription", "label"]].rename(
    columns={{"transcription": "text"}}
))
val_ds = Dataset.from_pandas(val_df[["transcription", "label"]].rename(
    columns={{"transcription": "text"}}
))

train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=64)
val_ds = val_ds.map(tokenize_fn, batched=True, batch_size=64)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ── Model ───────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

# ── Training Arguments ──────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
)

# ── Metrics ─────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted")
    return {{"f1_weighted": f1}}

# ── Train ───────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ── Evaluate ────────────────────────────────────────────────
eval_results = trainer.evaluate()
print(f"\\nEvaluation results: {{eval_results}}")

# Detailed classification report
preds = trainer.predict(val_ds)
pred_labels = np.argmax(preds.predictions, axis=-1)
true_labels = val_df["label"].values
print("\\nClassification Report:")
print(classification_report(
    true_labels, pred_labels,
    target_names=le.classes_,
    zero_division=0,
))

# ── Save ────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save label map
with open(f"{{OUTPUT_DIR}}/label_map.json", "w") as f:
    json.dump({{str(k): v for k, v in label_map.items()}}, f, indent=2)

# Save evaluation results
with open(f"{{OUTPUT_DIR}}/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\\n✅ Model saved to {{OUTPUT_DIR}}")
print("Download the fine_tuned_model/ folder and place it in models/transformer/")
'''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    logger.info("Kaggle training script written to %s", output_path)
