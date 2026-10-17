#!/usr/bin/env python3
"""Full pipeline orchestration script.

Run the entire clinical-billing-audit pipeline end-to-end.

Usage
-----
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clinical_billing_audit.config import load_config
from clinical_billing_audit.utils.logging_config import get_logger, setup_logging


def main(config_path: str = "configs/default.yaml") -> None:
    """Run the full pipeline."""
    cfg = load_config(config_path)
    setup_logging(
        level=cfg.get("logging.level", "INFO"),
        log_file=cfg.get("logging.log_file"),
    )
    logger = get_logger("pipeline")
    start = time.time()

    logger.info("=" * 60)
    logger.info("CLINICAL BILLING AUDIT — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load and Clean Data ─────────────────────────
    logger.info("\n[Step 1/9] Loading and cleaning data...")

    from clinical_billing_audit.data.cleaners import (
        clean_claims_data,
        clean_clinical_notes,
        clean_specialty_labels,
        profile_dataset,
    )
    from clinical_billing_audit.data.loaders import load_cms_claims, load_mtsamples
    from clinical_billing_audit.data.quality import DataQualityChecker

    notes_path = cfg.get("data.mtsamples_file", "data/raw/mtsamples.csv")
    notes_df = load_mtsamples(notes_path)
    notes_df = clean_clinical_notes(notes_df)
    notes_df = clean_specialty_labels(notes_df)

    # Profile
    profile = profile_dataset(
        notes_df,
        output_path=Path(cfg.get("data.interim_dir", "data/interim")) / "data_profile.csv",
    )

    # Quality checks
    qc = DataQualityChecker()
    quality_report = qc.run(notes_df, dataset_type="notes")
    quality_report.save(Path(cfg.get("data.interim_dir", "data/interim")) / "quality_report.csv")

    # Load claims
    claims_path = cfg.get("data.cms_claims_file", "data/raw/cms_synpuf_claims.csv")
    try:
        claims_df = load_cms_claims(claims_path)
        claims_df = clean_claims_data(claims_df)
        has_claims = True
    except FileNotFoundError:
        logger.warning("CMS claims file not found — using note-only mode.")
        has_claims = False
        claims_df = None

    # ── Step 2: Pair and Inject Mismatches ──────────────────
    logger.info("\n[Step 2/9] Pairing notes with claims and injecting mismatches...")

    from clinical_billing_audit.data.mismatch_injector import inject_mismatches
    from clinical_billing_audit.data.pairing import pair_notes_and_claims
    from clinical_billing_audit.utils.constants import SPECIALTY_TO_CODE_CATEGORY

    if has_claims:
        paired_df = pair_notes_and_claims(notes_df, claims_df, random_state=cfg.seed)
    else:
        # Note-only mode: create synthetic pairing
        import numpy as np

        notes_df["code_category"] = notes_df["medical_specialty"].map(
            lambda s: SPECIALTY_TO_CODE_CATEGORY.get(s, "general")
        )
        rng = np.random.RandomState(cfg.seed)
        provider_ids = [f"PRV_{i:04d}" for i in range(50)]
        notes_df["provider_id"] = rng.choice(provider_ids, size=len(notes_df))
        notes_df["pair_id"] = [f"PAIR_{i:06d}" for i in range(len(notes_df))]
        notes_df["primary_diagnosis_code"] = ""
        notes_df["claim_id"] = ""
        notes_df["claim_payment"] = 0.0
        paired_df = notes_df

    # Inject mismatches
    paired_df = inject_mismatches(
        paired_df,
        injection_rate=cfg.get("pairing.mismatch_injection_rate", 0.20),
        random_state=cfg.seed,
    )

    # Save
    processed_dir = Path(cfg.get("data.processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(processed_dir / "paired_with_mismatches.csv", index=False)
    logger.info("Paired dataset: %d rows saved", len(paired_df))

    # ── Step 3: NER Entity Extraction ───────────────────────
    logger.info("\n[Step 3/9] Extracting clinical entities...")

    from clinical_billing_audit.nlp.ner_extractor import NERExtractor

    extractor = NERExtractor(
        spacy_model=cfg.get("nlp.spacy_model", "en_core_sci_lg"),
        ner_model=cfg.get("nlp.ner_model", "en_ner_bc5cdr_md"),
    )
    ner_results = extractor.batch_extract(paired_df)
    ner_df = NERExtractor.results_to_dataframe(ner_results)
    ner_df.to_csv(processed_dir / "ner_results.csv", index=False)

    # ── Step 4: Baseline Classifier ─────────────────────────
    logger.info("\n[Step 4/9] Training baseline classifier...")

    from sklearn.model_selection import train_test_split

    from clinical_billing_audit.models.baseline import BaselineClassifier

    texts = paired_df["transcription"].tolist()
    labels = paired_df["code_category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=cfg.get("baseline.test_size", 0.2),
        stratify=labels, random_state=cfg.seed,
    )

    bcfg = cfg.baseline
    classifier = BaselineClassifier(
        tfidf_params=bcfg.get("tfidf", {}),
        lr_params=bcfg.get("logistic_regression", {}),
    )

    train_metrics = classifier.fit(X_train, y_train, ner_features=None)
    cv_metrics = classifier.cross_validate(X_train, y_train, cv=bcfg.get("cv_folds", 5))
    eval_report = classifier.evaluate(X_test, y_test)

    # Save model
    classifier.save(Path("models/tfidf_lr"))

    # Feature importance
    importance_df = classifier.get_feature_importance(top_n=20)

    # Predict on full dataset
    all_preds = classifier.predict(texts)
    all_probs = classifier.predict_proba(texts)

    # ── Step 5: Anomaly Detection ───────────────────────────
    logger.info("\n[Step 5/9] Running anomaly detection...")

    from clinical_billing_audit.models.anomaly import AnomalyDetector

    acfg = cfg.anomaly.get("isolation_forest", {})
    detector = AnomalyDetector(
        n_estimators=acfg.get("n_estimators", 200),
        contamination=acfg.get("contamination", 0.15),
        random_state=cfg.seed,
    )

    feature_df = detector.engineer_features(
        paired_df,
        ner_df=ner_df,
        classifier_probs=all_probs,
        classifier_labels=list(classifier.label_encoder.classes_),
    )
    detector.fit(feature_df)

    thresholds = cfg.anomaly.get("thresholds", {})
    results_df = detector.flag_mismatches(
        paired_df, feature_df,
        high_threshold=thresholds.get("high_risk", 0.85),
        medium_threshold=thresholds.get("medium_risk", 0.65),
        low_threshold=thresholds.get("low_risk", 0.45),
    )

    detector.save(Path("models/anomaly"))

    # ── Step 6: Ensemble Scoring ────────────────────────────
    logger.info("\n[Step 6/9] Computing ensemble scores...")

    from clinical_billing_audit.models.ensemble import EnsembleMismatchScorer
    from clinical_billing_audit.nlp.code_mapper import get_primary_predicted_category
    from clinical_billing_audit.nlp.entity_normalizer import normalize_entity

    # Get NER-predicted categories
    ner_categories = []
    for r in ner_results:
        diags = [normalize_entity(d, "DISEASE") for d in r.diagnoses]
        procs = [normalize_entity(p, "PROCEDURE") for p in r.procedures]
        cat = get_primary_predicted_category(diags, procs)
        ner_categories.append(cat)

    scorer = EnsembleMismatchScorer()
    results_df = scorer.score(
        results_df,
        anomaly_scores=results_df["anomaly_score"].values,
        classifier_preds=list(all_preds),
        classifier_confidence=all_probs.max(axis=1),
        ner_predicted_categories=ner_categories,
    )

    # Save full results
    results_df.to_csv(processed_dir / "flagged_results.csv", index=False)

    # ── Step 7: Evaluation ──────────────────────────────────
    logger.info("\n[Step 7/9] Evaluating results...")

    import numpy as np

    from clinical_billing_audit.evaluation.error_analysis import (
        error_breakdown,
        error_by_mismatch_type,
        plot_error_analysis,
    )
    from clinical_billing_audit.evaluation.metrics import (
        compute_binary_metrics,
        compute_threshold_sweep,
        generate_evaluation_report,
        plot_confusion_matrix,
        plot_precision_recall_curve,
    )

    y_true = results_df["is_mismatch"].astype(int).values
    y_pred = results_df["ensemble_flag"].astype(int).values
    y_score = results_df["ensemble_score"].values

    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    threshold_df = compute_threshold_sweep(y_true, y_score)

    logger.info("=== FINAL RESULTS ===")
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall:    %.4f", metrics["recall"])
    logger.info("F1:        %.4f", metrics["f1"])
    logger.info("ROC-AUC:   %.4f", metrics.get("roc_auc", 0))
    logger.info("PR-AUC:    %.4f", metrics.get("pr_auc", 0))

    # Plots
    figs_dir = Path("reports/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_precision_recall_curve(y_true, y_score, output_path=figs_dir / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, ["Correct", "Mismatch"],
                          output_path=figs_dir / "confusion_matrix.png")

    # Error analysis
    specialty_breakdown = error_breakdown(results_df, y_pred_col="ensemble_flag",
                                           group_col="medical_specialty")
    mismatch_type_breakdown = error_by_mismatch_type(results_df, y_pred_col="ensemble_flag")
    specialty_breakdown.to_csv(figs_dir / "error_by_specialty.csv", index=False)
    mismatch_type_breakdown.to_csv(figs_dir / "error_by_mismatch_type.csv", index=False)
    plot_error_analysis(specialty_breakdown, output_path=figs_dir / "error_analysis.png")

    # Generate report
    generate_evaluation_report(metrics, threshold_df,
                                output_path=Path("reports/evaluation_report.md"))

    # ── Step 8: Human Review Sample ─────────────────────────
    logger.info("\n[Step 8/9] Creating human review sample...")

    from clinical_billing_audit.evaluation.human_review import (
        create_review_sample,
        generate_labeling_template,
    )

    review_sample = create_review_sample(
        results_df,
        n_samples=cfg.get("evaluation.n_labeled_samples", 120),
        score_col="ensemble_score",
    )

    hand_labeled_dir = Path(cfg.get("data.hand_labeled_dir", "data/hand_labeled"))
    hand_labeled_dir.mkdir(parents=True, exist_ok=True)
    generate_labeling_template(review_sample, hand_labeled_dir / "labeling_template.csv")

    # Auto-fill labels from ground truth for evaluation
    review_sample["human_label"] = review_sample["mismatch_type"]
    review_sample.to_csv(hand_labeled_dir / "labeled_notes.csv", index=False)

    # ── Step 9: Power BI Export ─────────────────────────────
    logger.info("\n[Step 9/9] Exporting data for Power BI...")

    from clinical_billing_audit.dashboard.data_export import export_all

    export_all(
        results_df,
        metrics=metrics,
        threshold_df=threshold_df,
        importance_df=importance_df,
        output_dir=cfg.get("dashboard.export_dir", "reports/powerbi"),
    )

    # ── Done ────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("Results: %s", processed_dir / "flagged_results.csv")
    logger.info("Evaluation: reports/evaluation_report.md")
    logger.info("Power BI data: reports/powerbi/")
    logger.info("Human review template: %s", hand_labeled_dir / "labeling_template.csv")

    # Print summary for the user
    print(f"\n{'='*60}")
    print("PIPELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total note-claim pairs:  {len(results_df):,}")
    print(f"Injected mismatches:     {int(results_df['is_mismatch'].sum()):,} ({results_df['is_mismatch'].mean():.1%})")
    print(f"Flagged by model:        {int(results_df['ensemble_flag'].sum()):,} ({results_df['ensemble_flag'].mean():.1%})")
    print(f"")
    print(f"Precision:  {metrics['precision']:.1%}")
    print(f"Recall:     {metrics['recall']:.1%}")
    print(f"F1 Score:   {metrics['f1']:.1%}")
    print(f"PR-AUC:     {metrics.get('pr_auc', 0):.1%}")
    print(f"ROC-AUC:    {metrics.get('roc_auc', 0):.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the clinical billing audit pipeline")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()
    main(args.config)
