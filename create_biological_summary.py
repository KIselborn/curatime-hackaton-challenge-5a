#!/usr/bin/env python3
"""
Generate biological summaries linking SHAP features to CVD mechanisms.

This script:
1. Reads annotated SHAP reports (with taxonomy)
2. Groups OTUs by CVD mechanism (TMAO, Akkermansia, Butyrate, etc.)
3. Creates summary tables showing top contributors per mechanism
4. Generates HTML/markdown reports for easy visualization

Usage:
    python create_biological_summary.py \
        --shap-report models/gai_shap_analysis/gai_shap_analysis_annotated.json \
        --output models/gai_shap_analysis/biological_summary.html
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def create_biological_summary(shap_report_path, output_path=None):
    """
    Create biological summary from annotated SHAP report.
    
    Args:
        shap_report_path: Path to annotated SHAP report JSON
        output_path: Path to save HTML/CSV summary (default: same dir with _summary suffix)
    
    Returns:
        dict with summary statistics
    """
    
    if output_path is None:
        stem = Path(shap_report_path).stem
        parent = Path(shap_report_path).parent
        output_path = parent / f"{stem}_biological_summary.html"
    
    print(f"📖 Creating biological summary from: {shap_report_path}")
    
    # Load annotated report
    with open(shap_report_path, "r") as f:
        report = json.load(f)
    
    # Group features by mechanism
    mechanisms = defaultdict(list)
    for feat in report.get("annotated_features", []):
        otu_id = feat.get("otu_id", "unknown")
        readable_tax = feat.get("taxonomy", "Unknown")
        mean_shap = feat.get("mean_|shap|", 0)
        mechanism = feat.get("mechanism", "unclassified")
        
        mechanisms[mechanism].append({
            "otu_id": otu_id,
            "taxonomy": readable_tax,
            "mean_shap_abs": mean_shap,
            "direction": feat.get("direction", "unknown")
        })
    
    # Sort by mean |SHAP| within each mechanism
    for mech in mechanisms:
        mechanisms[mech].sort(key=lambda x: x["mean_shap_abs"], reverse=True)
    
    # Create summary tables
    summary_data = {}
    for mechanism, features in mechanisms.items():
        df = pd.DataFrame(features)
        summary_data[mechanism] = df
        print(f"\n📊 {mechanism.upper()}")
        print(f"   Features: {len(features)}")
        print("   Top 3:")
        for i, row in df.head(3).iterrows():
            print(f"     {i+1}. OTU {row['otu_id']:>8} | SHAP: {row['mean_shap_abs']:>6.3f} | {row['taxonomy']}")
    
    # Generate HTML report
    html_content = generate_html_report(mechanisms, report)
    
    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)
    
    # Also save CSV for each mechanism
    for mechanism, df in summary_data.items():
        csv_path = output_path.parent / f"mechanism_{mechanism}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   Saved: {csv_path}")
    
    print(f"\n✅ Biological summary saved to: {output_path}")
    
    return {
        "mechanisms": {k: len(v) for k, v in mechanisms.items()},
        "html_report": str(output_path),
        "summary_tables": {k: str(output_path.parent / f"mechanism_{k}.csv") for k in mechanisms.keys()}
    }


def generate_html_report(mechanisms, report):
    """Generate HTML report of mechanisms and their features."""
    
    mechanism_descriptions = {
        "tmao_producers": {
            "name": "TMAO Producers",
            "description": "Produce trimethylamine (TMA) from dietary choline; associated with CVD risk",
            "color": "#e74c3c",
            "taxa": "Clostridium, Ruminococcus, Prevotella, Alistipes"
        },
        "akkermansia": {
            "name": "Akkermansia & Barrier Function",
            "description": "Maintain intestinal barrier integrity; loss associated with inflammation",
            "color": "#3498db",
            "taxa": "Verrucomicrobia"
        },
        "butyrate_producers": {
            "name": "Butyrate Producers",
            "description": "Produce short-chain fatty acids (SCFAs); anti-inflammatory, protective",
            "color": "#2ecc71",
            "taxa": "Faecalibacterium, Roseburia, Anaerostipes"
        },
        "lps_producers": {
            "name": "LPS Producers",
            "description": "Gram-negative bacteria; lipopolysaccharide triggers inflammation",
            "color": "#f39c12",
            "taxa": "Proteobacteria, Escherichia, Klebsiella"
        },
        "bile_acid_metabolizers": {
            "name": "Bile Acid Metabolizers",
            "description": "Modify bile acids; affect lipid metabolism and immunity",
            "color": "#9b59b6",
            "taxa": "Bacteroides, Parabacteroides, Clostridium"
        },
        "diversity_indicator": {
            "name": "Diversity Indicators",
            "description": "Features representing overall microbiome richness and evenness",
            "color": "#1abc9c",
            "taxa": "Composite (Shannon, Chao1, evenness)"
        },
        "unclassified": {
            "name": "Unclassified",
            "description": "OTUs not matching predefined CVD patterns",
            "color": "#95a5a6",
            "taxa": "N/A"
        }
    }
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CVD & GAI SHAP Biological Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .mechanism-card {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-left: 5px solid;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #ecf0f1; font-weight: bold; }
            tr:hover { background: #f9f9f9; }
            .shap-bar {
                display: inline-block;
                height: 20px;
                background: #3498db;
                border-radius: 3px;
                min-width: 50px;
            }
            .high { background: #e74c3c; }
            .medium { background: #f39c12; }
            .low { background: #2ecc71; }
            .stats { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }
            code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🧬 CVD and Gut Aging Index (GAI) SHAP Biological Summary</h1>
        <p><em>Interpreting microbiome-driven prediction models through known CVD mechanisms</em></p>
    """
    
    # Add overall stats
    total_features = len(report.get("annotated_features", []))
    html += f"""
    <div class="stats">
        <strong>📊 Report Statistics:</strong><br>
        Total features analyzed: {total_features}<br>
        Base value (mean prediction): {report.get('base_value', 'N/A')}<br>
        Model type: {report.get('model_type', 'N/A')}
    </div>
    """
    
    # Add mechanism cards
    for mechanism_id, features in mechanisms.items():
        desc = mechanism_descriptions.get(mechanism_id, {
            "name": mechanism_id.replace("_", " ").title(),
            "description": "Mechanism details not available",
            "color": "#95a5a6",
            "taxa": "N/A"
        })
        
        html += f"""
        <div class="mechanism-card" style="border-color: {desc['color']}">
            <h2 style="color: {desc['color']}">{desc['name']}</h2>
            <p><strong>Description:</strong> {desc['description']}</p>
            <p><strong>Key Taxa:</strong> <code>{desc['taxa']}</code></p>
            <p><strong>Features contributing to this mechanism:</strong> {len(features)}</p>
            
            <table>
                <tr>
                    <th>OTU ID</th>
                    <th>Taxonomy</th>
                    <th>Mean |SHAP|</th>
                    <th>Direction</th>
                </tr>
        """
        
        for feat in features[:10]:  # Top 10 per mechanism
            shap_val = feat["mean_shap_abs"]
            direction = feat.get("direction", "neutral")
            color = "#e74c3c" if direction == "increases_prediction" else "#2ecc71" if direction == "decreases_prediction" else "#95a5a6"
            
            html += f"""
                <tr>
                    <td><code>{feat['otu_id']}</code></td>
                    <td>{feat['taxonomy']}</td>
                    <td><div class="shap-bar" style="width: {min(shap_val*50, 200)}px; background: {color}"></div> {shap_val:.4f}</td>
                    <td>{direction.replace('_', ' ').title()}</td>
                </tr>
            """
        
        if len(features) > 10:
            html += f"""
                <tr style="background: #ecf0f1;">
                    <td colspan="4"><em>... and {len(features)-10} more features</em></td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    html += """
        <hr>
        <footer style="color: #7f8c8d; font-size: 12px; margin-top: 30px;">
            <p>Generated by SHAP Biological Summary Tool</p>
            <p>For details on each mechanism, see the individual CSV files in the output directory.</p>
        </footer>
    </body>
    </html>
    """
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate biological summary from annotated SHAP reports")
    parser.add_argument(
        "--shap-report",
        type=Path,
        default=Path("models/gai_shap_analysis/gai_shap_analysis_annotated.json"),
        help="Path to annotated SHAP report"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file path"
    )
    
    args = parser.parse_args()
    
    if not args.shap_report.exists():
        raise FileNotFoundError(f"SHAP report not found: {args.shap_report}")
    
    result = create_biological_summary(args.shap_report, args.output)
    print(f"\n✅ Summary generation complete!")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
