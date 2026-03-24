"""
Load trained CVD model and generate SHAP-based interpretability analysis.
Can be used to re-analyze results or apply to new test sets.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.shap_analysis import run_shap_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP analysis for trained CVD prediction model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved LGBMClassifier (.joblib)"
    )
    parser.add_argument(
        "--test-features-path",
        type=str,
        required=True,
        help="Path to test feature matrix (CSV/TSV with features as columns)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for SHAP analysis (default: './shap_analysis_output')"
    )
    parser.add_argument(
        "--check-additivity",
        action="store_true",
        help="Validate SHAP value additivity (slower but validates correctness)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load test features
    features_path = Path(args.test_features_path)
    if not features_path.exists():
        print(f"Error: Test features file not found: {features_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading test features from {features_path}...")
    if features_path.suffix in ['.csv', '.tsv']:
        sep = '\t' if features_path.suffix == '.tsv' else ','
        X_test = pd.read_csv(features_path, sep=sep, index_col=0)
    else:
        print(f"Error: Unsupported file format: {features_path.suffix}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Test features shape: {X_test.shape}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("./shap_analysis_output")
    
    # Run SHAP analysis
    try:
        run_shap_analysis(
            model=model,
            X_test=X_test,
            output_dir=output_dir,
            check_additivity=args.check_additivity
        )
        
        print("\n✓ SHAP analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nKey outputs:")
        print(f"  • Report: {output_dir}/shap_analysis_report.json")
        print(f"  • Summary Plot: {output_dir}/shap_summary.png")
        print(f"  • Dependence Plots: {output_dir}/shap_dependence/")
        
    except Exception as e:
        print(f"Error during SHAP analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
