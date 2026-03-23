import pandas as pd
from pathlib import Path


DIAGNOSED_CVD_STRING = "Diagnosed by a medical professional (doctor, physician assistant)"

def load_processed_data(data_dir: str, dataset: str = "AGP"):
    """
    Load pre-processed OTU and metadata.
    Returns: meta, otu
    """
    base_path = Path(data_dir) / "processed" / dataset
    
    meta = pd.read_csv(base_path / "meta.tsv", sep="\t", index_col="id")
    otu = pd.read_csv(base_path / "otu.tsv", sep="\t", index_col="id")
    
    return meta, otu

def extract_cvd_labels(meta: pd.DataFrame, cvd_column: str = "cardiovascular_disease"):
    """
    Extract CVD labels from raw metadata since processed metadata only has general health flag.
    Returns: meta DataFrame with a new binary 'cvd' column.
    """
    if cvd_column not in meta.columns:
        raise ValueError(f"Missing required column '{cvd_column}' in metadata.")
    if "health" not in meta.columns:
        raise ValueError("Missing required 'health' column in metadata.")

    is_cvd = meta[cvd_column] == DIAGNOSED_CVD_STRING
    is_healthy = meta["health"] == "y"
    keep_mask = is_cvd | is_healthy

    meta_filtered = meta.loc[keep_mask].copy()
    meta_filtered["cvd"] = is_cvd.loc[keep_mask].astype(int)
    return meta_filtered


def load_agp_cvd_dataset(data_dir: str):
    """
    Load AGP processed data and join raw cardiovascular diagnosis labels.
    Returns: metadata_with_cvd, otu_aligned, y_binary
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
    labeled_meta = extract_cvd_labels(merged, cvd_column="cardiovascular_disease")
    otu = otu.loc[labeled_meta.index]
    y = labeled_meta["cvd"].astype(int)

    return labeled_meta, otu, y
