"""Command-line interface for the clinical-billing-audit pipeline.

Usage
-----
    cba data-quality --config configs/default.yaml
    cba extract-entities --config configs/default.yaml
    cba train --model baseline --config configs/default.yaml
    cba detect-anomalies --config configs/default.yaml
    cba evaluate --config configs/default.yaml
    cba run-all --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path

import click

from clinical_billing_audit.config import load_config
from clinical_billing_audit.utils.logging_config import setup_logging


@click.group()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to config YAML")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, config: str, verbose: bool) -> None:
    """Clinical Billing Audit — FWA Triage Pipeline."""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj["config"] = cfg

    level = "DEBUG" if verbose else cfg.get("logging.level", "INFO")
    log_file = cfg.get("logging.log_file")
    setup_logging(level=level, log_file=log_file)


@main.command()
@click.pass_context
def data_quality(ctx: click.Context) -> None:
    """Run data quality checks on raw data."""
    from clinical_billing_audit.data.cleaners import (
        profile_dataset,
    )
    from clinical_billing_audit.data.loaders import load_mtsamples
    from clinical_billing_audit.data.quality import DataQualityChecker
    from clinical_billing_audit.utils.logging_config import get_logger

    logger = get_logger("cli")
    cfg = ctx.obj["config"]

    logger.info("=== Data Quality Check ===")

    # Load and clean
    notes_path = cfg.get("data.mtsamples_file", "data/raw/mtsamples.csv")
    df = load_mtsamples(notes_path)

    # Profile
    profile = profile_dataset(
        df,
        output_path=Path(cfg.get("data.interim_dir", "data/interim")) / "data_profile.csv",
    )
    logger.info("Profile: %d rows, %d columns", profile["n_rows"], profile["n_columns"])

    # Quality checks
    qc = DataQualityChecker(
        max_null_fraction=cfg.get("quality.max_null_fraction", 0.10),
        max_duplicate_fraction=cfg.get("quality.max_duplicate_fraction", 0.05),
        min_note_length=cfg.get("quality.min_note_length", 50),
        max_note_length=cfg.get("quality.max_note_length", 50000),
    )
    report = qc.run(df, dataset_type="notes")
    report.save(Path(cfg.get("data.interim_dir", "data/interim")) / "quality_report.csv")

    if report.all_passed:
        click.echo(click.style("✅ All quality checks passed!", fg="green"))
    else:
        click.echo(click.style(f"⚠️  {report.n_failed} quality check(s) failed.", fg="yellow"))


@main.command()
@click.pass_context
def extract_entities(ctx: click.Context) -> None:
    """Extract clinical entities using NER."""
    from clinical_billing_audit.data.cleaners import clean_clinical_notes, clean_specialty_labels
    from clinical_billing_audit.data.loaders import load_mtsamples
    from clinical_billing_audit.nlp.ner_extractor import NERExtractor
    from clinical_billing_audit.utils.logging_config import get_logger

    logger = get_logger("cli")
    cfg = ctx.obj["config"]

    logger.info("=== NER Entity Extraction ===")
    df = load_mtsamples(cfg.get("data.mtsamples_file"))
    df = clean_clinical_notes(df)
    df = clean_specialty_labels(df)

    extractor = NERExtractor(
        spacy_model=cfg.get("nlp.spacy_model", "en_core_sci_lg"),
        ner_model=cfg.get("nlp.ner_model", "en_ner_bc5cdr_md"),
    )
    results = extractor.batch_extract(df)
    ner_df = NERExtractor.results_to_dataframe(results)

    output_path = Path(cfg.get("data.interim_dir", "data/interim")) / "ner_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ner_df.to_csv(output_path, index=False)
    click.echo(f"✅ Extracted entities for {len(ner_df)} notes → {output_path}")


@main.command()
@click.option("--model", "-m", type=click.Choice(["baseline", "transformer"]), default="baseline")
@click.pass_context
def train(ctx: click.Context, model: str) -> None:
    """Train a classifier model."""
    from clinical_billing_audit.data.cleaners import clean_clinical_notes, clean_specialty_labels
    from clinical_billing_audit.data.loaders import load_mtsamples
    from clinical_billing_audit.utils.constants import SPECIALTY_TO_CODE_CATEGORY
    from clinical_billing_audit.utils.logging_config import get_logger

    logger = get_logger("cli")
    cfg = ctx.obj["config"]

    logger.info("=== Training %s model ===", model)
    df = load_mtsamples(cfg.get("data.mtsamples_file"))
    df = clean_clinical_notes(df)
    df = clean_specialty_labels(df)
    df["code_category"] = df["medical_specialty"].map(
        lambda s: SPECIALTY_TO_CODE_CATEGORY.get(s, "general")
    )

    texts = df["transcription"].tolist()
    labels = df["code_category"].tolist()

    if model == "baseline":
        from sklearn.model_selection import train_test_split

        from clinical_billing_audit.models.baseline import BaselineClassifier

        bcfg = cfg.baseline
        classifier = BaselineClassifier(
            tfidf_params=bcfg.get("tfidf", {}),
            lr_params=bcfg.get("logistic_regression", {}),
        )

        # Split
        test_size = bcfg.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=cfg.seed,
        )

        # Train
        train_metrics = classifier.fit(X_train, y_train)
        click.echo(f"Training F1: {train_metrics['train_f1_weighted']:.4f}")

        # Evaluate
        eval_report = classifier.evaluate(X_test, y_test)
        click.echo(
            f"Test weighted F1: {eval_report.get('weighted avg', {}).get('f1-score', 0):.4f}"
        )

        # Save
        save_dir = Path("models/tfidf_lr")
        classifier.save(save_dir)
        click.echo(f"✅ Baseline model saved → {save_dir}")

    elif model == "transformer":
        from clinical_billing_audit.models.transformer import generate_kaggle_training_script

        output = Path("notebooks/kaggle_transformer_training.py")
        generate_kaggle_training_script(
            output,
            model_name=cfg.get("transformer.model_name", "dmis-lab/biobert-base-cased-v1.2"),
            num_epochs=cfg.get("transformer.num_epochs", 5),
            batch_size=cfg.get("transformer.batch_size", 16),
            learning_rate=cfg.get("transformer.learning_rate", 2e-5),
        )
        click.echo(f"✅ Kaggle training script generated → {output}")
        click.echo("Upload to Kaggle, enable GPU, and run!")


@main.command()
@click.pass_context
def detect_anomalies(ctx: click.Context) -> None:
    """Run anomaly detection on paired data."""
    from clinical_billing_audit.utils.logging_config import get_logger

    logger = get_logger("cli")
    cfg = ctx.obj["config"]

    logger.info("=== Anomaly Detection ===")

    processed_path = (
        Path(cfg.get("data.processed_dir", "data/processed")) / "paired_with_mismatches.csv"
    )
    if not processed_path.exists():
        click.echo(f"❌ Processed data not found at {processed_path}. Run the pipeline first.")
        return

    import pandas as pd

    from clinical_billing_audit.models.anomaly import AnomalyDetector

    df = pd.read_csv(processed_path)
    acfg = cfg.anomaly.get("isolation_forest", {})
    detector = AnomalyDetector(
        n_estimators=acfg.get("n_estimators", 200),
        contamination=acfg.get("contamination", 0.15),
        random_state=acfg.get("random_state", 42),
    )

    features = detector.engineer_features(df)
    detector.fit(features)
    results = detector.flag_mismatches(df, features)

    output = Path(cfg.get("data.processed_dir", "data/processed")) / "flagged_results.csv"
    results.to_csv(output, index=False)
    detector.save(Path("models/anomaly"))
    click.echo(f"✅ Anomaly detection complete → {output}")


@main.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Evaluate model against ground truth."""
    from clinical_billing_audit.utils.logging_config import get_logger

    logger = get_logger("cli")
    cfg = ctx.obj["config"]

    logger.info("=== Evaluation ===")

    results_path = Path(cfg.get("data.processed_dir", "data/processed")) / "flagged_results.csv"
    if not results_path.exists():
        click.echo(f"❌ Results not found at {results_path}. Run anomaly detection first.")
        return

    import pandas as pd

    from clinical_billing_audit.evaluation.metrics import (
        compute_binary_metrics,
        compute_threshold_sweep,
        generate_evaluation_report,
        plot_confusion_matrix,
        plot_precision_recall_curve,
    )

    df = pd.read_csv(results_path)

    if "is_mismatch" not in df.columns:
        click.echo("❌ No ground-truth labels found. Ensure mismatch injection was run.")
        return

    y_true = df["is_mismatch"].astype(int).values
    y_pred = df["is_flagged"].astype(int).values
    y_score = df.get("anomaly_score", pd.Series(0.5, index=df.index)).values

    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    threshold_df = compute_threshold_sweep(y_true, y_score)

    click.echo(f"\nPrecision: {metrics['precision']:.4f}")
    click.echo(f"Recall:    {metrics['recall']:.4f}")
    click.echo(f"F1:        {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        click.echo(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        click.echo(f"PR-AUC:    {metrics['pr_auc']:.4f}")

    # Save plots
    figs_dir = Path("reports/figures")
    plot_precision_recall_curve(y_true, y_score, output_path=figs_dir / "pr_curve.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=["Correct", "Mismatch"],
        output_path=figs_dir / "confusion_matrix.png",
    )

    # Save report
    generate_evaluation_report(
        metrics, threshold_df, output_path=Path("reports/evaluation_report.md")
    )
    click.echo("\n✅ Evaluation report saved → reports/evaluation_report.md")


@main.command(name="export-powerbi")
@click.option(
    "--output-dir", "-o", default="reports/powerbi", help="Output directory for Power BI CSVs"
)
@click.pass_context
def export_powerbi(ctx: click.Context, output_dir: str) -> None:
    """Export pipeline results as structured CSVs for Power BI."""
    import pandas as pd

    from clinical_billing_audit.dashboard.data_export import export_all

    cfg = ctx.obj["config"]
    processed_dir = Path(cfg.get("data.processed_dir", "data/processed"))
    results_path = processed_dir / "flagged_results.csv"

    if not results_path.exists():
        click.echo(f"❌ Results file not found at {results_path}. Run pipeline first.")
        return

    results_df = pd.read_csv(results_path)
    export_all(results_df=results_df, output_dir=output_dir)
    click.echo(f"✅ Power BI data exported to {output_dir}")


@main.command(name="run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run the complete pipeline end-to-end."""
    click.echo("🚀 Running full pipeline...")
    ctx.invoke(data_quality)
    ctx.invoke(extract_entities)
    ctx.invoke(train, model="baseline")
    ctx.invoke(detect_anomalies)
    ctx.invoke(evaluate)
    click.echo("🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
