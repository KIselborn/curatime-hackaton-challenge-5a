#!/usr/bin/env python3
"""
Standalone script to annotate existing SHAP reports with taxonomy and CVD mechanisms.

This script takes a previously generated SHAP report and enriches it with:
- Bacterial taxonomy names (readable phylum/class/order/family/genus format)
- CVD mechanism labels (TMAO, Akkermansia, butyrate, LPS, bile acid, diversity)
- Links between OTU IDs and their biological roles

Usage:
    python annotate_shap_with_taxonomy.py \
        --shap-report models/gai_shap_analysis/gai_shap_analysis_report.json \
        --taxonomy 97_otu_taxonomy.txt \
        --output models/gai_shap_analysis/gai_shap_analysis_annotated.json

Or with defaults (assumes standard structure):
    python annotate_shap_with_taxonomy.py
"""

import argparse
import json
from pathlib import Path
from src.taxonomy_mapper import annotate_gai_shap_report


def main():
    parser = argparse.ArgumentParser(
        description="Annotate SHAP reports with bacterial taxonomy and CVD mechanisms"
    )
    parser.add_argument(
        "--shap-report",
        type=Path,
        default=Path("models/gai_shap_analysis/gai_shap_analysis_report.json"),
        help="Path to SHAP analysis report JSON (default: models/gai_shap_analysis/gai_shap_analysis_report.json)"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("97_otu_taxonomy.txt"),
        help="Path to QIIME taxonomy file (default: 97_otu_taxonomy.txt in project root)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for annotated report (default: same as shap_report with _annotated suffix)"
    )
    parser.add_argument(
        "--cvd-shap",
        action="store_true",
        help="If set, annotate CVD SHAP report instead of GAI SHAP report"
    )
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        stem = args.shap_report.stem
        args.output = args.shap_report.parent / f"{stem}_annotated.json"
    
    print(f"📊 Annotating SHAP Report")
    print(f"  Input SHAP report: {args.shap_report}")
    print(f"  Taxonomy file: {args.taxonomy}")
    print(f"  Output path: {args.output}")
    
    # Validate files exist
    if not args.shap_report.exists():
        raise FileNotFoundError(f"SHAP report not found: {args.shap_report}")
    if not args.taxonomy.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {args.taxonomy}")
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Annotate the report
        result = annotate_gai_shap_report(
            gai_shap_report_path=args.shap_report,
            taxonomy_file=args.taxonomy,
            output_path=args.output
        )
        
        # Load and display summary
        with open(args.output, "r") as f:
            annotated = json.load(f)
        
        print(f"\n✅ Annotation successful!")
        print(f"\n📋 Summary:")
        print(f"  Total features annotated: {len(annotated.get('features_annotated', []))}")
        
        # Show top 5 annotated features
        if "features_annotated" in annotated:
            print(f"\n🔝 Top 5 annotated OTU features:")
            for i, feat in enumerate(annotated["features_annotated"][:5], 1):
                print(f"  {i}. OTU {feat['otu_id']}")
                print(f"     Taxonomy: {feat['readable_taxonomy']}")
                print(f"     Mechanism: {feat['biomarker_mechanism'].get('mechanism', 'unclassified')}")
                print(f"     Mean |SHAP|: {feat.get('mean_shap_abs', 'N/A')}")
        
        print(f"\n📁 Full annotated report: {args.output}")
        
    except Exception as e:
        print(f"❌ Annotation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
