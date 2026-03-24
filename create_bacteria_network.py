#!/usr/bin/env python3
"""
Create Network Visualization of Microbiome SHAP Analysis Results

This script generates an interactive network graph showing:
- Bacteria (OTUs) as nodes, labeled with most specific taxonomic names
- Node colors based on SHAP importance and CVD mechanism
- Node sizes based on SHAP values
- Edges representing relationships (correlation-based)

Usage:
    python create_bacteria_network.py --shap-report models/gai_shap_analysis/gai_shap_analysis_annotated.json --output bacteria_network.html
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns


def load_shap_data(shap_report_path):
    """Load SHAP analysis results with taxonomy."""
    with open(shap_report_path, 'r') as f:
        data = json.load(f)

    # Extract annotated features
    features = data.get('annotated_features', [])
    df = pd.DataFrame(features)

    # Add importance rank
    df = df.sort_values('mean_|shap|', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df, data


def get_most_specific_name(taxonomy_str):
    """Extract the most specific taxonomic name available."""
    if not taxonomy_str or taxonomy_str == 'Unknown':
        return 'Unknown'

    # Split taxonomy and find the deepest non-empty rank
    parts = taxonomy_str.split(' | ')
    for part in reversed(parts):
        if part and part not in ['???', 'Unknown', 'unclassified', 'uncultured']:
            return part

    return parts[0] if parts else 'Unknown'


def calculate_correlation_matrix(otu_data_path, top_otus):
    """Calculate correlation matrix for top OTUs."""
    try:
        # Load OTU data
        otu_df = pd.read_csv(otu_data_path, index_col=0)

        # Filter to top OTUs
        available_otus = [otu for otu in top_otus if otu in otu_df.columns]
        otu_subset = otu_df[available_otus]

        # Calculate correlation matrix
        corr_matrix = otu_subset.corr()

        return corr_matrix, available_otus

    except Exception as e:
        print(f"Warning: Could not calculate correlations: {e}")
        return None, top_otus


def create_network_graph(shap_df, corr_matrix=None, min_shap_threshold=0.05):
    """Create network graph from SHAP data."""

    # Filter to important features
    important_df = shap_df[shap_df['mean_|shap|'] >= min_shap_threshold].copy()

    # Create graph
    G = nx.Graph()

    # Color mapping for mechanisms
    mechanism_colors = {
        'tmao_producers': '#e74c3c',      # Red
        'butyrate_producers': '#27ae60',  # Green
        'bile_acid_metabolizers': '#3498db', # Blue
        'lps_producers': '#f39c12',       # Orange
        'barrier_protectors_akkermansia': '#9b59b6', # Purple
        'uncategorized': '#95a5a6'        # Gray
    }

    # Add nodes
    max_shap = important_df['mean_|shap|'].max()
    for _, row in important_df.iterrows():
        otu_id = row['otu_id']
        taxonomy = row['taxonomy']
        mechanism = row['mechanism']
        shap_value = row['mean_|shap|']

        # Get most specific name
        node_label = get_most_specific_name(taxonomy)

        # Node size based on SHAP importance (log scale for better visualization)
        node_size = 10 + (shap_value / max_shap) * 40

        # Node color based on mechanism
        node_color = mechanism_colors.get(mechanism, '#95a5a6')

        # Add node with attributes
        G.add_node(otu_id,
                  label=node_label,
                  title=f"{node_label}<br>OTU: {otu_id}<br>SHAP: {shap_value:.3f}<br>Mechanism: {mechanism}",
                  size=node_size,
                  color=node_color,
                  mechanism=mechanism,
                  shap_value=shap_value)

    # Add edges based on correlation (if available)
    if corr_matrix is not None:
        correlation_threshold = 0.3  # Minimum correlation for edge

        for i, otu1 in enumerate(corr_matrix.columns):
            for j, otu2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    corr = corr_matrix.loc[otu1, otu2]
                    if abs(corr) >= correlation_threshold:
                        # Edge weight based on correlation strength
                        edge_weight = abs(corr) * 3
                        G.add_edge(otu1, otu2, weight=edge_weight, correlation=corr)

    return G


def create_interactive_network(G, output_path, title="Microbiome SHAP Network"):
    """Create interactive network visualization."""

    # Create network
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#000000")

    # Add nodes
    for node, attrs in G.nodes(data=True):
        net.add_node(node,
                    label=attrs['label'],
                    title=attrs['title'],
                    size=attrs['size'],
                    color=attrs['color'])

    # Add edges
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1],
                    width=edge[2]['weight'],
                    title=f"Correlation: {edge[2]['correlation']:.3f}")

    # Configure physics
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      },
      "nodes": {
        "font": {
          "size": 12,
          "face": "arial"
        }
      },
      "edges": {
        "color": {
          "inherit": false,
          "color": "#cccccc"
        },
        "smooth": false
      }
    }
    """)

    # Add legend
    legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 12px;">
        <strong>Legend</strong><br>
        <span style="color: #e74c3c;">●</span> TMAO Producers<br>
        <span style="color: #27ae60;">●</span> Butyrate Producers<br>
        <span style="color: #3498db;">●</span> Bile Acid Metabolizers<br>
        <span style="color: #f39c12;">●</span> LPS Producers<br>
        <span style="color: #9b59b6;">●</span> Barrier Protectors<br>
        <span style="color: #95a5a6;">●</span> Uncategorized<br>
        <br>
        <em>Node size = SHAP importance<br>Edges = correlations > 0.3</em>
    </div>
    """

    # Save network
    net.save_graph(str(output_path))

    # Add legend to HTML
    with open(output_path, 'r') as f:
        html_content = f.read()

    # Insert legend before closing body tag
    html_content = html_content.replace('</body>', legend_html + '</body>')

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Interactive network saved to: {output_path}")


def create_static_network_plot(G, output_path):
    """Create static matplotlib network plot."""

    plt.figure(figsize=(15, 12))

    # Get positions
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]  # Scale up for visibility

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, edgecolors='black', linewidths=0.5)

    # Draw edges
    if G.edges():
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray')

    # Draw labels (only for top nodes to avoid clutter)
    labels = {}
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['shap_value'], reverse=True)
    for node in sorted_nodes[:15]:  # Label only top 15
        labels[node] = G.nodes[node]['label']

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title("Microbiome SHAP Network Analysis\n(Node size = SHAP importance, Colors = CVD mechanisms)",
             fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Static network plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create network visualization of microbiome SHAP analysis")
    parser.add_argument(
        "--shap-report",
        type=Path,
        required=True,
        help="Path to annotated SHAP report JSON"
    )
    parser.add_argument(
        "--otu-data",
        type=Path,
        default=None,
        help="Path to OTU abundance data for correlation analysis (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for network visualization"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Create interactive HTML network (default: static PNG)"
    )
    parser.add_argument(
        "--min-shap",
        type=float,
        default=0.05,
        help="Minimum SHAP value for node inclusion (default: 0.05)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top features to include (default: 50)"
    )

    args = parser.parse_args()

    print(f"🔬 Creating microbiome network from: {args.shap_report}")

    # Load SHAP data
    shap_df, metadata = load_shap_data(args.shap_report)

    # Filter to top N features
    shap_df = shap_df.head(args.top_n)

    print(f"📊 Processing {len(shap_df)} top features")

    # Calculate correlations if OTU data provided
    corr_matrix = None
    if args.otu_data and args.otu_data.exists():
        print("🔗 Calculating correlations between top OTUs...")
        top_otus = shap_df['otu_id'].tolist()
        corr_matrix, available_otus = calculate_correlation_matrix(args.otu_data, top_otus)
        if corr_matrix is not None:
            print(f"✓ Correlation matrix calculated for {len(available_otus)} OTUs")

    # Create network
    print("🌐 Building network graph...")
    G = create_network_graph(shap_df, corr_matrix, args.min_shap)

    print(f"📈 Network created with {len(G.nodes())} nodes and {len(G.edges())} edges")

    # Create visualization
    if args.interactive:
        create_interactive_network(G, args.output,
                                 title=f"Microbiome SHAP Network - {args.shap_report.stem}")
    else:
        create_static_network_plot(G, args.output)

    print("✅ Network visualization complete!")


if __name__ == "__main__":
    main()