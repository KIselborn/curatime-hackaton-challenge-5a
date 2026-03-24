#!/usr/bin/env python3
"""
Visualize important bacteria as a network graph.

Creates interactive network visualizations showing:
- OTU nodes colored by CVD mechanism/importance
- Node size proportional to SHAP impact
- Edges connecting related taxa
- Labels showing most specific bacterial names

Usage:
    python visualize_bacteria_network.py \
        --shap-report models/shap_analysis/shap_analysis_annotated.json \
        --output models/shap_analysis/bacteria_network.html

    python visualize_bacteria_network.py \
        --shap-report models/gai_shap_analysis/gai_shap_analysis_annotated.json \
        --output models/gai_shap_analysis/bacteria_network.html
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx
import plotly.graph_objects as go
import pandas as pd


# Color scheme for CVD mechanisms
MECHANISM_COLORS = {
    "tmao_producers": "#e74c3c",  # Red - ↑ CVD risk
    "lps_producers": "#c0392b",  # Dark red - ↑ CVD risk
    "barrier_protectors_akkermansia": "#2ecc71",  # Green - ↓ CVD risk
    "butyrate_producers": "#27ae60",  # Dark green - ↓ CVD risk
    "bile_acid_metabolizers": "#3498db",  # Blue - ↓ CVD risk
    "uncategorized": "#95a5a6"  # Gray - Unknown
}


def load_shap_report(report_path):
    """Load annotated SHAP report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def extract_most_specific_taxon(taxonomy_str):
    """Extract the most specific (last) ranked taxon from full taxonomy."""
    if "Unknown" in taxonomy_str or not taxonomy_str:
        return "Unknown"
    
    # Split by pipe and get the last non-empty part
    parts = [p.strip() for p in taxonomy_str.split("|")]
    for part in reversed(parts):
        if part and part != "???" and part not in ["Bacteria", "Archaea"]:
            return part
    
    return parts[-1] if parts else "Unknown"


def create_bacteria_network(shap_report, output_path=None):
    """
    Create network visualization of important bacteria.
    
    Args:
        shap_report: Loaded SHAP report dict
        output_path: Path to save HTML visualization
        
    Returns:
        networkx.Graph object
    """
    
    print("📊 Building bacteria network from SHAP results...")
    
    # Create graph
    G = nx.Graph()
    
    # Extract features from annotated report
    features = shap_report.get("annotated_features", [])
    
    if not features:
        print("⚠ No annotated features found in report!")
        return None
    
    # Add nodes with attributes
    node_positions = {}
    node_sizes = []
    node_colors = []
    node_labels = []
    node_hover = []
    
    max_shap = max([f.get("mean_|shap|", 0) for f in features])
    
    for idx, feature in enumerate(features):
        otu_id = str(feature.get("otu_id", f"OTU_{idx}"))
        taxonomy = feature.get("taxonomy", "Unknown")
        mechanism = feature.get("mechanism", "uncategorized")
        mean_shap = feature.get("mean_|shap|", 0)
        
        # Extract most specific name
        species_name = extract_most_specific_taxon(taxonomy)
        
        # Add node
        G.add_node(otu_id, 
                  taxonomy=taxonomy,
                  mechanism=mechanism,
                  shap_value=mean_shap,
                  species=species_name)
        
        node_labels.append(species_name)
        node_colors.append(MECHANISM_COLORS.get(mechanism, MECHANISM_COLORS["uncategorized"]))
        node_sizes.append(10 + (mean_shap / max_shap) * 50)  # Scale 10-60
        
        # Hover text
        direction = feature.get("direction", "")
        cvd_link = feature.get("cvd_link", "")
        hover_text = f"<b>{species_name}</b> ({otu_id})<br>" \
                    f"Taxonomy: {taxonomy}<br>" \
                    f"Mechanism: {mechanism}<br>" \
                    f"SHAP Impact: {mean_shap:.4f}<br>" \
                    f"Direction: {direction}<br>" \
                    f"CVD Link: {cvd_link}"
        node_hover.append(hover_text)
    
    # Add edges between related taxa (taxonomic similarity)
    # Connect OTUs that share family or higher-level taxonomy
    features_df = pd.DataFrame(features)
    
    for i in range(len(features)):
        for j in range(i + 1, min(i + 6, len(features))):  # Connect to next 5 similar features
            tax_i_parts = features[i]["taxonomy"].split("|")
            tax_j_parts = features[j]["taxonomy"].split("|")
            
            # Check if they share taxonomy levels (e.g., same family)
            shared_levels = sum([tax_i_parts[k].strip() == tax_j_parts[k].strip() 
                                for k in range(min(len(tax_i_parts), len(tax_j_parts)))])
            
            # Connect if they share 3+ taxonomic levels
            if shared_levels >= 3:
                otu_i = str(features[i]["otu_id"])
                otu_j = str(features[j]["otu_id"])
                G.add_edge(otu_i, otu_j, weight=shared_levels / 7.0)
    
    # If no edges were created, connect top features by SHAP value similarity
    if G.number_of_edges() == 0:
        sorted_features = sorted(features, key=lambda x: x.get("mean_|shap|", 0), reverse=True)
        for i in range(min(len(sorted_features) - 1, 10)):
            for j in range(i + 1, min(i + 4, len(sorted_features))):
                G.add_edge(str(sorted_features[i]["otu_id"]), 
                          str(sorted_features[j]["otu_id"]))
    
    print(f"✓ Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Prepare edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Prepare node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_hover,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='#444')
        ),
        showlegend=True,
        name="Bacteria"
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Add legend for mechanisms
    mechanism_traces = []
    for mechanism, color in MECHANISM_COLORS.items():
        mechanism_traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=mechanism.replace("_", " ").title(),
                hoverinfo='skip'
            )
        )
    
    for trace in mechanism_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Bacteria Network: CVD Risk Drivers & GAI Contributors</b><br><sup>Node size = SHAP impact | Colors = CVD mechanism</sup>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f9fa',
        height=800,
        width=1200
    )
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✓ Network visualization saved to {output_path}")
    
    return G, fig


def create_summary_stats(shap_report, G):
    """Generate summary statistics for the network."""
    
    print("\n📈 Network Statistics:")
    print(f"  Total nodes (bacteria): {G.number_of_nodes()}")
    print(f"  Total edges (connections): {G.number_of_edges()}")
    
    # Mechanism summary
    features = shap_report.get("annotated_features", [])
    mechanism_counts = defaultdict(int)
    mechanism_shap = defaultdict(float)
    
    for feature in features:
        mech = feature.get("mechanism", "uncategorized")
        mechanism_counts[mech] += 1
        mechanism_shap[mech] += feature.get("mean_|shap|", 0)
    
    print("\n🧬 Mechanism Breakdown:")
    for mechanism in sorted(mechanism_counts.keys(), key=lambda x: mechanism_shap[x], reverse=True):
        count = mechanism_counts[mechanism]
        shap_sum = mechanism_shap[mechanism]
        print(f"  {mechanism:30s}: {count:3d} OTUs (SHAP Σ = {shap_sum:.3f})")
    
    # Top OTUs
    print("\n🏆 Top 10 by SHAP Impact:")
    sorted_features = sorted(features, key=lambda x: x.get("mean_|shap|", 0), reverse=True)
    for i, feature in enumerate(sorted_features[:10], 1):
        species = extract_most_specific_taxon(feature.get("taxonomy", "Unknown"))
        shap_val = feature.get("mean_|shap|", 0)
        print(f"  {i:2d}. {species:40s} SHAP: {shap_val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bacteria network from SHAP analysis"
    )
    parser.add_argument(
        "--shap-report",
        type=Path,
        required=True,
        help="Path to annotated SHAP report JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for HTML visualization (default: same dir with _network.html)"
    )
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        stem = args.shap_report.stem
        args.output = args.shap_report.parent / f"{stem}_network.html"
    
    print(f"📊 Creating bacteria network visualization")
    print(f"  Input: {args.shap_report}")
    print(f"  Output: {args.output}")
    
    # Load report
    report = load_shap_report(args.shap_report)
    
    # Create network
    G, fig = create_bacteria_network(report, args.output)
    
    if G is not None:
        # Print statistics
        create_summary_stats(report, G)
        print(f"\n✅ Network visualization complete!")
        print(f"   Open: {args.output}")
    else:
        print("❌ Failed to create network")


if __name__ == "__main__":
    main()
