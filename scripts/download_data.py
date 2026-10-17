#!/usr/bin/env python3
"""Data download and synthetic sample generator for Clinical Billing Audit.

Supports:
1. Generating a realistic sample dataset (--sample / --synthetic) so the entire
   pipeline can be tested and demonstrated immediately without external downloads.
2. Downloading MTSamples from Kaggle using the Kaggle API.
3. Information and download guidance for CMS DE-SynPUF public use files.

Usage
-----
    # Generate realistic sample data for instant local demo / testing:
    python scripts/download_data.py --sample

    # Download from Kaggle (requires ~/.kaggle/kaggle.json):
    python scripts/download_data.py --kaggle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from clinical_billing_audit.utils.logging_config import get_logger, setup_logging

logger = get_logger("data_downloader")


# ── Sample Clinical Templates ──────────────────────────────────────────────
SAMPLE_NOTES = [
    # Cardiology / Circulatory
    (
        "Cardiovascular / Pulmonary",
        "Coronary Artery Disease and Angina",
        "HISTORY OF PRESENT ILLNESS: A 64-year-old male presents with worsening exertional chest pain radiating to the left arm over the past three weeks. Pain is described as pressure-like, relieved with sublingual nitroglycerin. Past medical history significant for essential hypertension and hyperlipidemia. MEDICATIONS: Lisinopril 20 mg daily, Atorvastatin 40 mg daily, Aspirin 81 mg daily. PHYSICAL EXAMINATION: BP 142/88 mmHg, HR 76 bpm. Cardiac exam reveals regular rate and rhythm, normal S1 and S2, no murmurs, rubs, or gallops. Lungs clear to auscultation bilaterally. ASSESSMENT: 1. Stable angina pectoris. 2. Coronary artery disease. 3. Essential hypertension. PLAN: Schedule diagnostic coronary angiography. Titrate lisinopril. Continue antiplatelet therapy.",
        "chest pain, coronary artery disease, angina, hypertension, lisinopril, nitroglycerin",
    ),
    (
        "Cardiovascular / Pulmonary",
        "Congestive Heart Failure Exacerbation",
        "CHIEF COMPLAINT: Shortness of breath and bilateral lower extremity edema. HISTORY OF PRESENT ILLNESS: A 72-year-old female with known systolic heart failure (EF 30%) presents with a 5-day history of orthopnea, paroxysmal nocturnal dyspnea, and an 8-pound weight gain. MEDICATIONS: Furosemide 40 mg daily, Carvedilol 12.5 mg twice daily, Sacubitril/Valsartan. EXAM: BP 155/92 mmHg, HR 88 bpm. Elevated JVP at 8 cm. Bibasilar crackles on lung exam. 2+ pitting edema to mid-shin bilaterally. S3 gallop present. IMPRESSION: Acute decompensated congestive heart failure. PLAN: Admit to telemetry. IV Furosemide 80 mg bolus, monitor strict intake and output, repeat BNP and comprehensive metabolic panel.",
        "heart failure, dyspnea, orthopnea, edema, furosemide, carvedilol, bnp",
    ),
    # Orthopedic / Musculoskeletal
    (
        "Orthopedic",
        "Left Knee Osteoarthritis - Arthroscopy Consultation",
        "SUBJECTIVE: The patient is a 58-year-old female with longstanding severe left knee joint pain, unresponsive to physical therapy and intra-articular corticosteroid injections. Pain is worse with weight bearing. OBJECTIVE: Antalgic gait favoring the right leg. Left knee examination shows moderate joint effusion, tenderness along the medial joint line, and positive McMurray test for medial meniscus tear. Range of motion is limited to 10-105 degrees. Radiographs demonstrate severe medial joint space narrowing with subchondral sclerosis and osteophyte formation. DIAGNOSIS: 1. Severe primary osteoarthritis of left knee. 2. Degenerative tear of medial meniscus. PLAN: Discussed options including arthroscopic debridement versus total knee arthroplasty.",
        "knee pain, osteoarthritis, joint space narrowing, meniscus tear, arthroscopy, total knee arthroplasty",
    ),
    (
        "Orthopedic",
        "Right Rotator Cuff Tear",
        "HISTORY: 52-year-old male laborer reporting severe right shoulder pain after a lifting incident at work two months ago. Inability to abduct the arm past 90 degrees. MRI of right shoulder shows a full-thickness tear of the supraspinatus tendon measuring 2.2 cm with mild retraction. EXAM: Positive empty can test, positive drop arm sign, positive Hawkins impingement. Neurovascular examination intact distally. ASSESSMENT: Complete tear of right rotator cuff, impingement syndrome. PLAN: Patient scheduled for right shoulder arthroscopic rotator cuff repair with subacromial decompression.",
        "shoulder pain, rotator cuff tear, supraspinatus, subacromial decompression, arthroscopy",
    ),
    # Gastroenterology / Digestive
    (
        "Gastroenterology",
        "Screening and Diagnostic Colonoscopy",
        "PROCEDURE PERFORMED: Colonoscopy with snare polypectomy. INDICATION: Routine screening and iron deficiency anemia in a 61-year-old male. PROCEDURE DESCRIPTION: The Olympus adult colonoscope was introduced through the rectum and advanced to the cecum under moderate sedation with propofol. Examination of the ascending colon revealed a 1.2 cm sessile polyp, which was completely resected using a cold snare and retrieved for histology. In the sigmoid colon, moderate diverticulosis without diverticulitis was observed. The terminal ileum appeared normal. DIAGNOSIS: 1. Colon polyp, completely removed. 2. Diverticulosis coli. RECOMMENDATION: Await pathology; repeat colonoscopy in 3 to 5 years based on histology.",
        "colonoscopy, polypectomy, polyp, diverticulosis, anemia, screening",
    ),
    (
        "Gastroenterology",
        "Gastroesophageal Reflux Disease and Esophagitis",
        "CLINICAL SUMMARY: 45-year-old male with persistent retrosternal burning and dysphagia despite over-the-counter famotidine. Upper endoscopy (EGD) was performed under conscious sedation. Esophagus showed linear mucosal breaks greater than 5 mm extending between the tops of the mucosal folds (Los Angeles Grade B reflux esophagitis). Stomach and duodenum demonstrated normal mucosa with negative rapid urease test for H. pylori. IMPRESSION: 1. Erosive reflux esophagitis, LA Grade B. 2. GERD. PLAN: Initiate Omeprazole 40 mg daily before breakfast. Lifestyle and dietary modifications reviewed.",
        "gerd, esophagitis, dysphagia, endoscopy, omeprazole, famotidine",
    ),
    # Neurology / Nervous
    (
        "Neurology",
        "Acute Migraine with Aura and Status Migrainosus",
        "SUBJECTIVE: 34-year-old female with recurrent episodic migraine who presents with a continuous throbbing left hemicranial headache lasting 84 hours, accompanied by photophobia, phonophobia, and visual scintillating scotoma. Failed home rescue therapy with sumatriptan and naproxen. NEUROLOGICAL EXAM: Alert, oriented x 4. Cranial nerves II-XII grossly intact. No focal motor weakness, normal sensation, no meningismus or Kernig sign. Gait normal. ASSESSMENT: Status migrainosus without intractable aura. PLAN: Administered IV hydration, IV ketorolac 30 mg, metoclopramide 10 mg, and diphenhydramine. Prescribed topiramate for migraine prophylaxis.",
        "migraine, headache, status migrainosus, sumatriptan, photophobia, topiramate",
    ),
    (
        "Neurology",
        "Transient Ischemic Attack Evaluation",
        "HISTORY: 68-year-old male with history of diabetes and smoking who experienced sudden transient right-sided facial droop and expressive aphasia lasting approximately 25 minutes, with complete spontaneous resolution. Brain MRI with DWI shows no acute restricted diffusion or acute infarction. Carotid duplex demonstrates 60% stenosis of the left internal carotid artery. EKG in sinus rhythm. DIAGNOSIS: Transient ischemic attack (TIA) in left MCA territory. PLAN: Patient admitted for observation. Initiated Dual Antiplatelet Therapy (DAPT) with aspirin and clopidogrel. High-intensity statin therapy started. Cardiology consult for Holter monitoring.",
        "tia, transient ischemic attack, carotid stenosis, clopidogrel, aspirin, mri",
    ),
    # General Surgery / Surgical
    (
        "Surgery",
        "Laparoscopic Cholecystectomy for Cholelithiasis",
        "PREOPERATIVE DIAGNOSIS: Symptomatic cholelithiasis with biliary colic. POSTOPERATIVE DIAGNOSIS: Chronic cholecystitis with multiple gallstones. PROCEDURE: Laparoscopic cholecystectomy. FINDINGS: The gallbladder was chronically inflamed with thick walls containing multiple pigmented stones. Cystic duct and cystic artery were meticulously dissected, clipped with hem-o-lok clips, and divided. Critical view of safety was confirmed. Gallbladder removed via endobag through umbilical port. Estimated blood loss < 25 mL. Sponge and needle counts correct. Patient tolerated procedure well.",
        "laparoscopic cholecystectomy, cholelithiasis, gallstones, cholecystitis, critical view of safety",
    ),
    (
        "Surgery",
        "Laparoscopic Appendectomy",
        "OPERATIVE REPORT: Patient is a 24-year-old female presenting to the emergency room with acute periumbilical abdominal pain migrating to the right lower quadrant, fever (38.4 C), and leukocytosis (WBC 14,800). CT abdomen showed enlarged appendix measuring 10 mm with appendicolith and periappendiceal fat stranding. Laparoscopic appendectomy was performed through 3-port technique. Inflamed, turgid appendix identified, mesoappendix divided with harmonic scalpel, and base secured with endoloop. Specimen sent to pathology. Postoperative recovery uneventful.",
        "appendectomy, appendicitis, abdominal pain, laparoscopy, leukocytosis",
    ),
    # Urology / Genitourinary
    (
        "Urology",
        "Nephrolithiasis with Ureteral Stent Placement",
        "INDICATION: 48-year-old male with acute left flank pain radiating to the groin with microscopic hematuria. Non-contrast CT demonstrated a 6 mm obstructing left distal ureteral stone with moderate hydronephrosis. PROCEDURE PERFORMED: Cystoscopy, left retrograde pyelogram, and insertion of double-J ureteral stent. FINDINGS: Obstructing calculus encountered at the left ureterovesical junction. Guide wire passed successfully. 6 French x 26 cm double-J stent deployed with excellent proximal and distal curl. Bladder mucosa normal. DIAGNOSIS: Obstructing left ureteral calculus, hydronephrosis. PLAN: Stented for decompression; outpatient laser lithotripsy scheduled next week.",
        "urology, nephrolithiasis, kidney stone, ureteral stent, hydronephrosis, cystoscopy",
    ),
    # Dermatology / Skin
    (
        "Dermatology",
        "Excision of Malignant Melanoma",
        "CLINICAL PRESENTATION: 56-year-old male with changing pigmented lesion on right upper back showing asymmetry, irregular borders, and color variegation. Previous punch biopsy demonstrated malignant melanoma, Breslow depth 0.8 mm, Clark level III, no ulceration. PROCEDURE: Wide local excision with 1.0 cm surgical margins down to deep fascia, followed by intermediate layered closure. Bleeding controlled with electrocautery. Specimen oriented and sent for permanent margin evaluation. DIAGNOSIS: Cutaneous malignant melanoma, trunk. PLAN: Wound care instructions provided. Suture removal in 14 days. Routine lymph node surveillance.",
        "melanoma, skin lesion, wide local excision, breslow depth, dermatology, biopsy",
    ),
]


def generate_synthetic_mtsamples(
    output_path: Path,
    n_samples: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic MTSamples dataset."""
    rng = np.random.RandomState(random_state)
    records = []

    # Common variations and patient age/sex injectors
    ages = range(20, 85)
    sexes = ["male", "female"]

    for i in range(n_samples):
        template = SAMPLE_NOTES[i % len(SAMPLE_NOTES)]
        specialty, sample_name, transcription, keywords = template

        # Add minor natural variability
        age = rng.choice(ages)
        sex = rng.choice(sexes)
        custom_note = transcription.replace("64-year-old male", f"{age}-year-old {sex}")
        custom_note = custom_note.replace("72-year-old female", f"{age}-year-old {sex}")
        custom_note = custom_note.replace("58-year-old female", f"{age}-year-old {sex}")
        custom_note = custom_note.replace("52-year-old male", f"{age}-year-old {sex}")

        records.append({
            "description": f"{sample_name} clinical consultation and record.",
            "medical_specialty": specialty,
            "sample_name": f"{sample_name} #{i + 1}",
            "transcription": custom_note,
            "keywords": keywords,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d synthetic MTSamples notes to %s", len(df), output_path)
    return df


def generate_synthetic_cms_claims(
    output_path: Path,
    n_claims: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic CMS DE-SynPUF style inpatient/outpatient claims."""
    rng = np.random.RandomState(random_state)

    # Common ICD-9 codes aligned with specialties
    icd9_options = [
        # Circulatory
        ("4019", "41401", "4280", 12500.0),
        ("42731", "4280", "4019", 9800.0),
        # Musculoskeletal
        ("71590", "7140", "7172", 18200.0),
        ("72761", "72610", "71941", 8900.0),
        # Digestive
        ("2113", "56210", "2809", 4200.0),
        ("53081", "53010", "78703", 3100.0),
        # Nervous
        ("34690", "7840", "30000", 2500.0),
        ("4359", "4019", "25000", 7800.0),
        # Surgical / Digestive
        ("57420", "57410", "5759", 14300.0),
        ("5409", "78900", "78060", 11200.0),
        # Genitourinary
        ("5921", "591", "7881", 6400.0),
        # Skin / Neoplasm
        ("1725", "2382", "7099", 5100.0),
    ]

    records = []
    providers = [f"PRV_{i:04d}" for i in range(1, 41)]

    for i in range(n_claims):
        diag_choice = icd9_options[i % len(icd9_options)]
        prm_diag = diag_choice[0]
        sec_diag = diag_choice[1]
        base_pmt = diag_choice[3]
        actual_pmt = round(float(rng.normal(base_pmt, base_pmt * 0.15)), 2)
        if actual_pmt < 500:
            actual_pmt = 500.0

        records.append({
            "DESYNPUF_ID": f"BENE_{rng.randint(10000, 99999):05d}",
            "CLM_ID": f"CLM_{i:07d}",
            "SEGMENT": 1,
            "CLM_FROM_DT": 20080101 + rng.randint(0, 360),
            "CLM_THRU_DT": 20080110 + rng.randint(0, 360),
            "PRVDR_NUM": rng.choice(providers),
            "CLM_PMT_AMT": actual_pmt,
            "NCH_PRMRY_PYR_CLM_PD_AMT": 0.0,
            "AT_PHYSN_NPI": f"NPI_{rng.randint(100000, 999999)}",
            "OP_PHYSN_NPI": f"NPI_{rng.randint(100000, 999999)}",
            "OT_PHYSN_NPI": "",
            "CLM_ADMSN_DT": 20080101 + rng.randint(0, 360),
            "ADMTNG_ICD9_DGNS_CD": prm_diag,
            "CLM_PASS_THRU_PER_DIEM_AMT": 0.0,
            "NCH_BENE_IP_DDCTBL_AMT": 1024.0,
            "NCH_BENE_PTA_COINSRNC_LBLTY_AM": 0.0,
            "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM": 0.0,
            "CLM_UTLZTN_AMT": round(actual_pmt * 1.25, 2),
            "PRVDR_STATE_CD": rng.choice(["CA", "NY", "TX", "FL", "IL", "PA"]),
            "ICD9_DGNS_CD_1": prm_diag,
            "ICD9_DGNS_CD_2": sec_diag,
            "ICD9_DGNS_CD_3": diag_choice[2],
            "ICD9_PRCDR_CD_1": f"{rng.randint(30, 89):02d}{rng.randint(1, 9)}",
            "ICD9_PRCDR_CD_2": "",
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d synthetic CMS claims to %s", len(df), output_path)
    return df


def generate_hand_labeled_dataset(
    output_path: Path,
    n_samples: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic hand-labeled review dataset for evaluation benchmarks."""
    rng = np.random.RandomState(random_state)
    labels = ["correct_match", "upcoding", "miscoding", "undercoding"]
    weights = [0.70, 0.15, 0.10, 0.05]  # Natural clinical audit distribution

    records = []
    for i in range(n_samples):
        label = rng.choice(labels, p=weights)
        is_mismatch = 1 if label != "correct_match" else 0
        specialty = rng.choice([
            "Cardiovascular / Pulmonary",
            "Orthopedic",
            "Gastroenterology",
            "Neurology",
            "Surgery",
            "Urology",
        ])

        if label == "correct_match":
            reviewer_notes = "Clinical note and procedure documentation strictly align with billed code category."
            confidence = 0.95
        elif label == "upcoding":
            reviewer_notes = "Billed for high-complexity surgical procedure; operative note describes standard diagnostic exploration."
            confidence = 0.90
        elif label == "miscoding":
            reviewer_notes = "Claim billed under gastroenterology; clinical note records an acute orthopedic evaluation."
            confidence = 0.92
        else:
            reviewer_notes = "Billed minimal code, but documented work supported a higher complexity evaluation."
            confidence = 0.85

        records.append({
            "pair_id": f"PAIR_{i:06d}",
            "note_id": f"MT_{i:05d}",
            "medical_specialty": specialty,
            "human_label": label,
            "is_mismatch": is_mismatch,
            "reviewer_confidence": confidence,
            "reviewer_id": f"AUDITOR_{rng.choice([1, 2, 3])}",
            "audit_date": "2026-10-15",
            "reviewer_notes": reviewer_notes,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d hand-labeled audit records to %s", len(df), output_path)
    return df


def download_kaggle_mtsamples(output_dir: Path) -> None:
    """Attempt to download MTSamples using Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_dir / "mtsamples.csv"
    if target_file.exists():
        logger.info("MTSamples file already exists at %s", target_file)
        return

    logger.info("Attempting download from Kaggle: tboyle10/medicaltranscriptions...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "tboyle10/medicaltranscriptions", "-p", str(output_dir), "--unzip"],
            check=True,
        )
        logger.info("Successfully downloaded MTSamples from Kaggle.")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(
            "Kaggle CLI download failed or not configured (%s).\n"
            "To use real data:\n"
            "  1. Ensure kaggle is installed: pip install kaggle\n"
            "  2. Place API token in ~/.kaggle/kaggle.json\n"
            "  3. Run: kaggle datasets download -d tboyle10/medicaltranscriptions -p data/raw --unzip\n"
            "Alternatively, run this script with --sample to generate realistic sample data instantly.",
            e,
        )


def main() -> None:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(description="Download or generate data for Clinical Billing Audit")
    parser.add_argument("--sample", "--synthetic", action="store_true", help="Generate realistic synthetic sample data")
    parser.add_argument("--kaggle", action="store_true", help="Attempt to download MTSamples via Kaggle CLI")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--hand-labeled-dir", default="data/hand_labeled", help="Path to hand-labeled data directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    hand_dir = Path(args.hand_labeled_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    hand_dir.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_kaggle_mtsamples(raw_dir)

    # Always ensure sample datasets exist if raw files are absent or --sample is flagged
    mtsamples_path = raw_dir / "mtsamples.csv"
    claims_path = raw_dir / "cms_synpuf_claims.csv"
    hand_labeled_path = hand_dir / "labeled_notes.csv"

    if args.sample or not mtsamples_path.exists():
        generate_synthetic_mtsamples(mtsamples_path, n_samples=1000)

    if args.sample or not claims_path.exists():
        generate_synthetic_cms_claims(claims_path, n_claims=1200)

    if args.sample or not hand_labeled_path.exists():
        generate_hand_labeled_dataset(hand_labeled_path, n_samples=120)

    logger.info("Data preparation completed.")
    logger.info("MTSamples: %s", mtsamples_path)
    logger.info("CMS Claims: %s", claims_path)
    logger.info("Hand-labeled: %s", hand_labeled_path)


if __name__ == "__main__":
    main()
