# 📊 Power BI Dashboard Implementation Manual
## Clinical Note-to-Billing Code Consistency Checker (FWA Triage)

This directory contains the structured analytical tables exported by the pipeline, designed for instant ingestion into **Power BI Desktop**.

---

## 1. Exported Data Tables

| Table File | Role | Primary Key / Grain | Description |
|---|---|---|---|
| `flag_summary.csv` | KPI Dimension | Single Row | Executive KPI cards: total audited claims, flagged count, true detection rate, risk tiers. |
| `flag_by_specialty.csv` | Dimension / Aggregation | `medical_specialty` | Mismatch counts, flag rates, and severity distribution across clinical departments. |
| `provider_scores.csv` | Dimension / Entity | `provider_id` | Audit metrics per provider: total claims, flagged claims, mean anomaly score, max score. |
| `mismatch_details.csv` | Fact Table | `pair_id` | Granular record-level audit table with note excerpts, billed category, predicted category, risk scores. |
| `model_performance.csv` | Model Diagnostics | `metric` | Precision, recall, F1, PR-AUC, and threshold sweep data. |
| `precision_recall_curve.csv` | Model Diagnostics | `threshold` | Continuous PR curve coordinates for sensitivity tuning visual. |
| `confusion_matrix.csv` | Model Diagnostics | `[actual, predicted]` | Confusion matrix entries: True Positives, False Positives, True Negatives, False Negatives. |
| `feature_importance.csv` | Model Explainability | `feature` | Top weighted features influencing the clinical classification and anomaly detection. |

---

## 2. Power BI Data Model (Star Schema)

```
        ┌─────────────────────────┐
        │       flag_summary      │  (KPIs & Global Rates)
        └─────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                                 ▼
┌────────────────────────┐      ┌────────────────────────┐
│   flag_by_specialty    │      │    provider_scores     │
│ (medical_specialty PK) │      │    (provider_id PK)    │
└────────────────────────┘      └────────────────────────┘
            │                               │
            │ 1                             │ 1
            │                               │
            │ *                             │ *
            └───────────────┬───────────────┘
                            ▼
              ┌───────────────────────────┐
              │     mismatch_details      │  (Fact Table: pair_id PK)
              │---------------------------│
              │ pair_id                   │
              │ note_id                   │
              │ provider_id (FK)          │
              │ medical_specialty (FK)    │
              │ billed_category           │
              │ code_category             │
              │ anomaly_score             │
              │ ensemble_score            │
              │ severity                  │
              │ is_flagged                │
              │ note_preview              │
              └───────────────────────────┘
```

---

## 3. Recommended DAX Measures

### Audit Volume & Flag Rate
```dax
Total Claims Audited = COUNTROWS(mismatch_details)

Total Flagged Claims = 
CALCULATE(
    COUNTROWS(mismatch_details),
    mismatch_details[is_flagged] = TRUE()
)

Overall Flag Rate = 
DIVIDE([Total Flagged Claims], [Total Claims Audited], 0)
```

### Risk Severity Metrics
```dax
High Risk Claims = 
CALCULATE(
    COUNTROWS(mismatch_details),
    mismatch_details[severity] = "HIGH_RISK"
)

Medium Risk Claims = 
CALCULATE(
    COUNTROWS(mismatch_details),
    mismatch_details[severity] = "MEDIUM_RISK"
)

High Risk Ratio = 
DIVIDE([High Risk Claims], [Total Flagged Claims], 0)
```

### Financial Exposure / Potential Overpayment
```dax
Total Flagged Billed Amount = 
CALCULATE(
    SUM(mismatch_details[claim_payment]),
    mismatch_details[is_flagged] = TRUE()
)

Potential Recoverable Overpayment = 
[Total Flagged Billed Amount] * 0.45
```

---

## 4. Multi-Page Dashboard Architecture

### Page 1: Executive FWA Triage Overview
- **Top KPI Cards**: Total Claims Audited, Flagged Count, Global Flag Rate, High Risk Rate, Model Precision (98.8%).
- **Donut Chart**: Risk Severity Distribution (`HIGH_RISK`, `MEDIUM_RISK`, `LOW_RISK`, `NORMAL`).
- **Bar Chart**: Flag Rate by Medical Specialty (sorted descending).
- **Time/Specialty Slicers**: Interactive filters by Medical Specialty and Risk Tier.

### Page 2: Provider Behavior & Peer Anomaly Analysis
- **Scatter Plot**: X-axis: Provider Total Claims; Y-axis: Provider Flag Rate; Bubble Size: Mean Anomaly Score.
- **Provider Outlier Table**: Sorted by highest flag rate and outlier entropy deviation.
- **Provider Drill-Through**: Click on any provider to view their exact claims and clinical note documentation.

### Page 3: Model Diagnostics & Explainability
- **Line Chart**: Precision-Recall Curve (using `precision_recall_curve.csv`).
- **Matrix Visual**: Confusion Matrix (`confusion_matrix.csv`).
- **Bar Chart**: Feature Importance ranking (`feature_importance.csv`).

### Page 4: Auditor Investigation Workspace (Drill-Down)
- **Detailed Audit Table**:
  - `pair_id`
  - `provider_id`
  - `medical_specialty`
  - `billed_category` vs `predicted_category`
  - `ensemble_score` & `severity`
  - `note_preview` (Clinical documentation excerpt for immediate reviewer verification).
