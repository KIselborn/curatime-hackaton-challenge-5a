import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def apply_clr_transform(otu: pd.DataFrame, pseudo_count=1):
    """
    Apply Centered Log-Ratio (CLR) transformation to compositional microbiome data.
    """
    # Add pseudo-count to handle zeros
    otu_pseudo = otu + pseudo_count
    
    # Calculate geometric mean for each sample
    geom_mean = np.exp(np.mean(np.log(otu_pseudo), axis=1))
    
    # Apply CLR: log(x / geom_mean)
    clr_transformed = np.log(otu_pseudo.div(geom_mean, axis=0))
    return clr_transformed

def filter_features_by_variance(otu: pd.DataFrame, threshold=0.01):
    """
    Remove features with extremely low variance.
    """
    variances = otu.var()
    keep_cols = variances[variances > threshold].index
    return otu[keep_cols]

def prepare_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data ensuring stratification.
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
