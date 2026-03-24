import pandas as pd
from pathlib import Path
import numpy as np


# --------------------------------------------------
# CONSTANT (AGP specific exact label)
# --------------------------------------------------
DIAGNOSED_CVD_STRING = "Diagnosed by a medical professional (doctor, physician assistant)"


# --------------------------------------------------
# LOAD PROCESSED DATA (COMMON)
# --------------------------------------------------
def load_processed_data(data_dir: str, dataset: str):
    """
    Load pre-processed OTU and metadata.
    Returns: meta, otu
    """
    base_path = Path(data_dir) / "processed" / dataset

    meta = pd.read_csv(base_path / "meta.tsv", sep="\t", index_col="id")
    otu = pd.read_csv(base_path / "otu.tsv", sep="\t", index_col="id")

    return meta, otu


# --------------------------------------------------
# AGP: EXACT MATCH LABEL EXTRACTION
# --------------------------------------------------
def extract_cvd_labels_agp(meta: pd.DataFrame, cvd_column: str = "cardiovascular_disease"):
    """
    Extract CVD labels using exact string match (AGP-specific).
    """

    if cvd_column not in meta.columns:
        raise ValueError(f"Missing required column '{cvd_column}'")

    is_cvd = meta[cvd_column] == DIAGNOSED_CVD_STRING

    meta_filtered = meta.copy()
    meta_filtered["cvd"] = is_cvd.fillna(False).astype(int)

    return meta_filtered


# --------------------------------------------------
# GGMP: KEYWORD-BASED LABEL EXTRACTION
# --------------------------------------------------
def extract_cvd_labels_gcmp(meta: pd.DataFrame, cvd_column: str):
    """
    Extract CVD labels using keyword matching (GGMP).
    """

    if cvd_column not in meta.columns:
        raise ValueError(f"Missing required column '{cvd_column}'")

    # keywords capturing cardiovascular diseases
    cvd_keywords = [
        "cardio",
        "heart",
        "stroke",
        "hypertension",
        "coronary",
        "artery",
        "infarction",
        "blood pressure",
    ]

    col = meta[cvd_column].astype(str).str.lower()

    is_cvd = col.apply(lambda x: any(k in x for k in cvd_keywords))

    meta_filtered = meta.copy()
    meta_filtered["cvd"] = is_cvd.fillna(False).astype(int)

    return meta_filtered


# --------------------------------------------------
# AGP LOADER
# --------------------------------------------------
def load_agp_cvd_dataset(data_dir: str):
    """
    Load AGP dataset with CVD labels.
    """

    processed_meta, otu = load_processed_data(data_dir, dataset="AGP")

    raw_meta_path = Path(data_dir) / "raw" / "AGP" / "AGP-metadata-feces.tsv"

    raw_meta = pd.read_csv(
        raw_meta_path,
        sep="\t",
        usecols=["#SampleID", "cardiovascular_disease"],
        index_col="#SampleID",
        low_memory=False,
    )

    merged = processed_meta.join(raw_meta, how="left")

    labeled_meta = extract_cvd_labels_agp(
        merged,
        cvd_column="cardiovascular_disease"
    )

    otu = otu.loc[labeled_meta.index]
    y = labeled_meta["cvd"].fillna(False).astype(int)

    return labeled_meta, otu, y


# --------------------------------------------------
# GGMP LOADER
# --------------------------------------------------
def load_gcmp_cvd_dataset(data_dir: str):
    """
    Load GGMP dataset with keyword-based CVD labels.
    """

    processed_meta, otu = load_processed_data(data_dir, dataset="GGMP")

    raw_meta_path = Path(data_dir) / "raw" / "GGMP" / "GGMP-metadata.tsv"

    raw_meta = pd.read_csv(
        raw_meta_path,
        sep="\t",
        index_col="#SampleID",
        low_memory=False,
    )

    # ---- auto-detect possible CVD column ----
    possible_cols = [
        "dis_atherosclerosis",
        "heart_angina_pectoris",
        "heart_bypass_surgery",
        "heart_stent_surgery",
        "stroke_hemorrhagic",
        "stroke_ischemic",
        "heart_aspirin",
        "heart_statin",
        "dis_MS(metabolic_syndrome)",
        "dis_T2D(diabetes)",
        "dis_fatty_liver",
    ]

    cvd_column = None
    for col in possible_cols:
        if col in raw_meta.columns:
            merged = processed_meta.join(raw_meta[[cvd_column]], how="left")

             # ---- label extraction ----
            labeled_meta = extract_cvd_labels_gcmp(
                  merged,
                cvd_column=col
            )
            
            if cvd_column is None:
                cvd_column= labeled_meta["cvd"]
            else:
                cvd_column = cvd_column + labeled_meta["cvd"]      
                
                cvd_column = np.clip(cvd_column,0,1)         
                

            

    if cvd_column is None:
        raise ValueError("No suitable CVD column found in GGMP metadata.")

    print(f"[GGMP] Using column for CVD detection: {cvd_column}")

   

    otu = otu.loc[labeled_meta.index]
    y = cvd_column.astype(int)

    return labeled_meta, otu, y