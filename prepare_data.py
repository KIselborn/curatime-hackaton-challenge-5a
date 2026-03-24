"""
Data Preparation Script for GAI (Gut Aging Index) Calculator

Processes BIOM files (OTU data) and supplementary metadata tables from the paper
into the format expected by gai_cal.py:
  - meta.tsv: columns [id, age, health] where health is 'y' or 'n'
  - otu.tsv:  columns [id, OTU_1, OTU_2, ...] with abundance values

Metadata source: Supplementary Excel (41598_2024_82418_MOESM2_ESM.xlsx)
  - Table S4: GGMP metadata with ~40 disease columns (7,009 samples)
  - Table S6: AGP metadata with pre-computed health column (5,966 samples)

Health criteria (from paper & supplementary):
  GGMP: No disease across ~40 columns, FBG < 6.1, BMI < 24, no antibiotics
  AGP:  Pre-computed by authors (12 disease columns + BMI < 24 + no antibiotics)

OTUs present in < 10% of the filtered population are excluded.

Requirements: pip install pandas numpy biom-format h5py scipy openpyxl
"""

import os
import numpy as np
import pandas as pd
from biom import load_table

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Downloaded-Data")
SUPP_DIR = os.path.join(BASE_DIR, "Supplementary-Material")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed-Data")

SUPP_EXCEL = os.path.join(SUPP_DIR, "41598_2024_82418_MOESM2_ESM.xlsx")
GGMP_BIOM_PATH = os.path.join(DATA_DIR, "GCMP", "GGMP-feces.biom")
AGP_BIOM_PATH = os.path.join(DATA_DIR, "AGP", "AGP-feces.biom")

OTU_PREVALENCE_THRESHOLD = 0.10

# ─── GGMP disease columns from Supplementary Table S4 ────────────────────────
# ~40 disease/condition columns that define "healthy" when all are negative.
# Grouped by category for clarity.

GGMP_HEART_STROKE_COLS = [
    "heart_bypass_surgery", "heart_stent_surgery", "heart_angina_pectoris",
    "heart_aspirin", "heart_statins", "stroke_ischemic", "stroke_hemorrhagic",
]
GGMP_RESPIRATORY_COLS = ["copd", "asthma"]
GGMP_GENERAL_DISEASE_COLS = [
    "osteoarticular_disease", "waist_neck_disease",
    "digestive_system_disease", "urinary_system_disease",
    # malignant_tumor_disease handled separately (coded 'a' = absent)
]
GGMP_SPECIFIC_DISEASE_COLS = [
    "dis_T1DM", "dis_T2DM", "dis_fatty_liver", "dis_psoriasis",
    "dis_AD", "dis_PD", "dis_ASD", "dis_MS",
    "dis_atherosclerosis", "dis_LE", "dis_ARDS", "dis_gastritis",
    "dis_hepatic_calculus", "dis_cholecystitis", "dis_colitis", "dis_IBS",
    "dis_kidneyStone", "dis_gout", "dis_AS", "dis_RA",
    "dis_neurosis", "dis_CFS", "dis_constipation_symptom", "dis_diarrhea_symptom",
]
# All binary (y/n) disease columns combined
GGMP_ALL_YN_DISEASE_COLS = (
    GGMP_HEART_STROKE_COLS + GGMP_RESPIRATORY_COLS +
    GGMP_GENERAL_DISEASE_COLS + GGMP_SPECIFIC_DISEASE_COLS +
    ["MetS"]  # Metabolic syndrome
)

# GGMP BIOM sample ID format: "11757.{SampleID}.56280"
GGMP_BIOM_PREFIX = "11757."
GGMP_BIOM_SUFFIX = ".56280"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def biom_to_filtered_dataframe(biom_path, sample_ids, prevalence_threshold):
    """
    Load a BIOM file, filter to given sample IDs, filter OTUs by prevalence,
    and return a samples x OTUs DataFrame.
    Operates on sparse matrices until the final step to save memory.
    """
    print(f"  Loading BIOM: {os.path.basename(biom_path)}")
    biom_table = load_table(biom_path)
    print(f"  Raw BIOM: {biom_table.shape[0]:,} OTUs x {biom_table.shape[1]:,} samples")

    # Filter to requested samples
    biom_sample_set = set(biom_table.ids(axis="sample"))
    common_ids = sorted(set(sample_ids) & biom_sample_set)
    n_missing = len(sample_ids) - len(common_ids)
    if n_missing > 0:
        print(f"  NOTE: {n_missing} sample IDs not found in BIOM")
    biom_table = biom_table.filter(common_ids, axis="sample", inplace=False)
    print(f"  After sample filter: {biom_table.shape[0]:,} OTUs x "
          f"{biom_table.shape[1]:,} samples")

    # OTU prevalence on sparse matrix (observations x samples)
    sparse_mat = biom_table.matrix_data
    n_samples = sparse_mat.shape[1]
    otu_presence = np.array((sparse_mat > 0).sum(axis=1)).flatten()
    prevalence = otu_presence / n_samples

    # Filter OTUs by prevalence
    keep_mask = prevalence >= prevalence_threshold
    keep_otu_ids = biom_table.ids(axis="observation")[keep_mask]
    biom_table = biom_table.filter(keep_otu_ids, axis="observation", inplace=False)
    print(f"  After OTU prevalence filter (>= {prevalence_threshold*100:.0f}%): "
          f"{biom_table.shape[0]:,} OTUs x {biom_table.shape[1]:,} samples")

    # Convert to dense DataFrame
    otu_df = pd.DataFrame(
        biom_table.matrix_data.toarray().T,
        index=biom_table.ids(axis="sample"),
        columns=biom_table.ids(axis="observation"),
    )
    otu_df.index.name = "id"
    return otu_df


def print_cohort_stats(meta_df, label, paper_total, paper_healthy, paper_nonhealthy):
    """Print cohort statistics and compare with paper-reported values."""
    n_total = len(meta_df)
    n_h = (meta_df["health"] == "y").sum()
    n_nh = (meta_df["health"] == "n").sum()
    age_h = meta_df.loc[meta_df["health"] == "y", "age"]
    age_nh = meta_df.loc[meta_df["health"] == "n", "age"]

    print(f"\n  {label} Cohort Summary:")
    print(f"    Total samples:  {n_total:>6,}  (paper: {paper_total:,})")
    print(f"    Healthy:        {n_h:>6,}  (paper: {paper_healthy:,})"
          f"  [{100*n_h/max(n_total,1):.1f}%]")
    print(f"    Non-healthy:    {n_nh:>6,}  (paper: {paper_nonhealthy:,})"
          f"  [{100*n_nh/max(n_total,1):.1f}%]")
    if len(age_h) > 0:
        print(f"    Healthy mean age:     {age_h.mean():.2f} +/- {age_h.std():.2f}"
              f"  (paper: {label == 'GGMP' and '45.97 +/- 16.38' or '45.43 +/- 14.91'})")
    if len(age_nh) > 0:
        print(f"    Non-healthy mean age: {age_nh.mean():.2f} +/- {age_nh.std():.2f}"
              f"  (paper: {label == 'GGMP' and '54.05 +/- 14.01' or '49.57 +/- 14.15'})")


# ─── GGMP Processing ─────────────────────────────────────────────────────────

def process_ggmp():
    """
    Process the GGMP (Guangdong Gut Microbiome Project) dataset.

    Uses Supplementary Table S4 for metadata (correct 40-disease definitions).
    Maps short SampleIDs (e.g. G440205594) to BIOM IDs (11757.G440205594.56280).

    Health criteria (paper + Table S4):
      - All ~38 binary disease columns (y/n) == 'n'
      - malignant_tumor_disease == 'a' (absent)
      - MetS (metabolic syndrome) == 'n'
      - FBG < 6.1  (biochem_FBG)
      - BMI < 24   (anthrop_BMI)
      - No antibiotics (antibiotics == 'n')
      - Complete phenotypic information
    """
    print("=" * 65)
    print("  GGMP Dataset Processing")
    print("=" * 65)

    # ── Load metadata from Supplementary Table S4 ─────────────────────────
    print("\n[1/4] Loading metadata from Supplementary Table S4...")
    meta_df = pd.read_excel(SUPP_EXCEL, sheet_name="Sup Table 4", header=1)
    meta_df["SampleID"] = meta_df["SampleID"].astype(str)
    print(f"  Table S4: {meta_df.shape[0]:,} samples x {meta_df.shape[1]} columns")

    # Convert numeric columns
    meta_df["age"] = pd.to_numeric(meta_df["age"], errors="coerce")
    meta_df["anthrop_BMI"] = pd.to_numeric(meta_df["anthrop_BMI"], errors="coerce")
    meta_df["biochem_FBG"] = pd.to_numeric(meta_df["biochem_FBG"], errors="coerce")

    # ── Filter for complete phenotypic data ───────────────────────────────
    print("\n[2/4] Filtering for complete phenotypic data...")

    available_yn = [c for c in GGMP_ALL_YN_DISEASE_COLS if c in meta_df.columns]
    print(f"  Disease columns (y/n): {len(available_yn)}")

    has_age = meta_df["age"].notna() & (meta_df["age"] >= 18)
    has_bmi = meta_df["anthrop_BMI"].notna()
    has_fbg = meta_df["biochem_FBG"].notna()
    has_antibiotics = meta_df["antibiotics"].isin(["y", "n"])
    has_tumor_info = meta_df["malignant_tumor_disease"].notna()
    has_disease_info = meta_df[available_yn].isin(["y", "n"]).all(axis=1)

    complete = has_age & has_bmi & has_fbg & has_antibiotics & has_tumor_info & has_disease_info
    meta_df = meta_df[complete].copy()
    print(f"  Samples with complete data: {len(meta_df):,}")

    # ── Define health status ──────────────────────────────────────────────
    print("\n[3/4] Defining health status...")

    # All y/n disease columns must be 'n'
    all_diseases_negative = (meta_df[available_yn] == "n").all(axis=1)
    # Malignant tumor: 'a' = absent
    no_tumor = meta_df["malignant_tumor_disease"] == "a"
    # Biochemical & anthropometric thresholds
    fbg_ok = meta_df["biochem_FBG"] < 6.1
    bmi_ok = meta_df["anthrop_BMI"] < 24
    # No antibiotics
    no_antibiotics = meta_df["antibiotics"] == "n"

    is_healthy = all_diseases_negative & no_tumor & fbg_ok & bmi_ok & no_antibiotics
    meta_df["health"] = np.where(is_healthy, "y", "n")

    print_cohort_stats(meta_df, "GGMP",
                       paper_total=6014, paper_healthy=1133, paper_nonhealthy=4881)

    # ── Map SampleIDs to BIOM format & load OTU data ──────────────────────
    print("\n[4/4] Loading BIOM and filtering OTUs...")

    # Table S4 SampleID: "G440205594" -> BIOM: "11757.G440205594.56280"
    biom_id_map = {
        GGMP_BIOM_PREFIX + sid + GGMP_BIOM_SUFFIX: sid
        for sid in meta_df["SampleID"]
    }
    biom_ids = list(biom_id_map.keys())

    otu_df = biom_to_filtered_dataframe(GGMP_BIOM_PATH, biom_ids, OTU_PREVALENCE_THRESHOLD)
    print(f"  (Paper reports: 942 OTUs after filtering)")

    # Build output with BIOM IDs as index (what gai_cal.py expects)
    common_biom_ids = otu_df.index.tolist()
    # Map back to get corresponding SampleIDs for metadata lookup
    reverse_map = {v: k for k, v in biom_id_map.items() if k in set(common_biom_ids)}

    meta_df = meta_df.set_index("SampleID")
    meta_out = meta_df.loc[reverse_map.keys(), ["age", "health"]].copy()
    meta_out["age"] = meta_out["age"].astype(int)
    # Replace index with BIOM IDs so it matches otu_df
    meta_out.index = [reverse_map[sid] for sid in meta_out.index]
    meta_out.index.name = "id"

    otu_out = otu_df.loc[meta_out.index].copy()

    print(f"\n  Final GGMP output: {len(meta_out):,} samples, {otu_out.shape[1]:,} OTUs")
    return meta_out, otu_out


# ─── AGP Processing ──────────────────────────────────────────────────────────

def process_agp():
    """
    Process the AGP (American Gut Project) dataset.

    Uses Supplementary Table S6 which contains the authors' pre-filtered
    5,966 samples with a pre-computed 'health' column (y/n).
    SampleIDs directly match BIOM file IDs.

    Health criteria applied by authors (paper + Table S6):
      - 12 disease columns: ibd, alzheimers, asd, autoimmune, cancer,
        cardiovascular_disease, diabetes, ibs, kidney_disease, liver_disease,
        lung_disease, mental_illness — all "I do not have this condition"
      - BMI < 24
      - No antibiotic use in past year
    """
    print("\n" + "=" * 65)
    print("  AGP Dataset Processing")
    print("=" * 65)

    # ── Load metadata from Supplementary Table S6 ─────────────────────────
    print("\n[1/3] Loading metadata from Supplementary Table S6...")
    meta_df = pd.read_excel(SUPP_EXCEL, sheet_name="Sup Table 6", header=1)
    meta_df["SampleID"] = meta_df["SampleID"].astype(str)
    print(f"  Table S6: {meta_df.shape[0]:,} samples x {meta_df.shape[1]} columns")

    meta_df["age"] = pd.to_numeric(meta_df["age"], errors="coerce")

    # Use the pre-computed health column from the authors
    print(f"  Health column values: {meta_df['health'].value_counts().to_dict()}")

    # Filter out any rows with missing age or health
    valid = meta_df["age"].notna() & meta_df["health"].isin(["y", "n"])
    meta_df = meta_df[valid].copy()

    print_cohort_stats(meta_df, "AGP",
                       paper_total=5966, paper_healthy=1852, paper_nonhealthy=4114)

    # ── Load BIOM & filter OTUs ───────────────────────────────────────────
    print("\n[2/3] Loading BIOM and filtering OTUs...")
    biom_ids = meta_df["SampleID"].tolist()
    otu_df = biom_to_filtered_dataframe(AGP_BIOM_PATH, biom_ids, OTU_PREVALENCE_THRESHOLD)

    # ── Align and output ──────────────────────────────────────────────────
    print("\n[3/3] Aligning metadata and OTU data...")
    common_ids = otu_df.index.intersection(meta_df.set_index("SampleID").index)
    meta_df = meta_df.set_index("SampleID")
    meta_out = meta_df.loc[common_ids, ["age", "health"]].copy()
    meta_out["age"] = meta_out["age"].astype(float)
    meta_out.index.name = "id"
    otu_out = otu_df.loc[common_ids].copy()
    otu_out.index.name = "id"

    print(f"\n  Final AGP output: {len(meta_out):,} samples, {otu_out.shape[1]:,} OTUs")
    return meta_out, otu_out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── GGMP ──────────────────────────────────────────────────────────────
    ggmp_dir = os.path.join(OUTPUT_DIR, "GGMP")
    os.makedirs(ggmp_dir, exist_ok=True)

    ggmp_meta, ggmp_otu = process_ggmp()
    ggmp_meta.to_csv(os.path.join(ggmp_dir, "meta.tsv"), sep="\t")
    ggmp_otu.to_csv(os.path.join(ggmp_dir, "otu.tsv"), sep="\t")
    print(f"\n  Saved: {ggmp_dir}/meta.tsv  ({ggmp_meta.shape[0]} x {ggmp_meta.shape[1]})")
    print(f"  Saved: {ggmp_dir}/otu.tsv   ({ggmp_otu.shape[0]} x {ggmp_otu.shape[1]})")

    # ── AGP ───────────────────────────────────────────────────────────────
    agp_dir = os.path.join(OUTPUT_DIR, "AGP")
    os.makedirs(agp_dir, exist_ok=True)

    agp_meta, agp_otu = process_agp()
    agp_meta.to_csv(os.path.join(agp_dir, "meta.tsv"), sep="\t")
    agp_otu.to_csv(os.path.join(agp_dir, "otu.tsv"), sep="\t")
    print(f"\n  Saved: {agp_dir}/meta.tsv  ({agp_meta.shape[0]} x {agp_meta.shape[1]})")
    print(f"  Saved: {agp_dir}/otu.tsv   ({agp_otu.shape[0]} x {agp_otu.shape[1]})")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    print("  To run the GAI pipeline:")
    print(f"    python gai_cal.py {ggmp_dir}/meta.tsv {ggmp_dir}/otu.tsv")
    print(f"    python gai_cal.py {agp_dir}/meta.tsv {agp_dir}/otu.tsv")
    print("=" * 65)


if __name__ == "__main__":
    main()
