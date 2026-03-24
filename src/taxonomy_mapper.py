"""
OTU ID to Bacteria Taxonomy Mapper
Connects OTU feature IDs to actual bacterial taxonomy and CVD-relevant mechanisms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# CVD-relevant taxa patterns (case-insensitive substring matching)
CVD_TAXA_PATTERNS = {
    "tmao_producers": {
        "pattern": ["Clostridium", "Ruminococcus", "Prevotella", "Alistipes"],
        "cvd_link": "TMAO production pathway (atherogenic metabolite)",
        "direction": "↑ abundance = ↑ CVD risk"
    },
    "barrier_protectors_akkermansia": {
        "pattern": ["Akkermansia", "Verrucomicrobia"],
        "cvd_link": "Intestinal barrier function & LPS reduction",
        "direction": "↓ abundance = ↑ CVD risk"
    },
    "butyrate_producers": {
        "pattern": ["Faecalibacterium", "Roseburia", "Lachnospiraceae"],
        "cvd_link": "SCFA production (anti-inflammatory, tight junction maintenance)",
        "direction": "↓ abundance = ↑ CVD risk"
    },
    "lps_producers": {
        "pattern": ["Proteobacteria", "Escherichia", "Klebsiella", "Enterobacteriaceae"],
        "cvd_link": "LPS production (pro-inflammatory endotoxemia)",
        "direction": "↑ abundance = ↑ CVD risk"
    },
    "bile_acid_metabolizers": {
        "pattern": ["Bacteroides", "Parabacteroides", "Clostridium"],
        "cvd_link": "Secondary bile acid metabolism (FXR/TGR5 signaling)",
        "direction": "↓ abundance = ↑ CVD risk"
    }
}


class TaxonomyMapper:
    """Map OTU IDs to bacterial taxonomy and CVD-relevant mechanisms."""
    
    def __init__(self, taxonomy_file: Path):
        """
        Initialize with taxonomy file.
        
        Args:
            taxonomy_file: Path to taxonomy TSV (OTU_ID \t k__...; p__...; etc.)
        """
        self.taxonomy_file = Path(taxonomy_file)
        self.otu_taxonomy = {}
        self.load_taxonomy()
    
    def load_taxonomy(self) -> None:
        """Load OTU ID → taxonomy mapping from file."""
        print(f"Loading taxonomy from {self.taxonomy_file}...")
        with open(self.taxonomy_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    otu_id = parts[0]
                    taxonomy = parts[1]
                    self.otu_taxonomy[otu_id] = taxonomy
        
        print(f"Loaded {len(self.otu_taxonomy):,} OTU taxonomy entries")
    
    def get_taxonomy(self, otu_id: str) -> Optional[str]:
        """
        Retrieve full taxonomy string for OTU ID.
        
        Args:
            otu_id: OTU identifier (as string)
            
        Returns:
            Taxonomy lineage string or None if not found
        """
        return self.otu_taxonomy.get(str(otu_id))
    
    def parse_taxonomy(self, taxonomy_str: str) -> Dict[str, str]:
        """
        Parse QIIME taxonomy string into dict.
        
        Args:
            taxonomy_str: e.g. "k__Bacteria; p__Firmicutes; c__Clostridia; ..."
            
        Returns:
            {rank: taxon, ...} e.g. {"kingdom": "Bacteria", "phylum": "Firmicutes", ...}
        """
        if not taxonomy_str:
            return {}
        
        rank_map = {
            "k__": "kingdom",
            "p__": "phylum",
            "c__": "class",
            "o__": "order",
            "f__": "family",
            "g__": "genus",
            "s__": "species"
        }
        
        result = {}
        for part in taxonomy_str.split("; "):
            part = part.strip()
            for prefix, rank in rank_map.items():
                if part.startswith(prefix):
                    taxon = part[len(prefix):].strip()
                    if taxon:  # Only include non-empty taxa
                        result[rank] = taxon
                    break
        
        return result
    
    def get_readable_taxonomy(self, otu_id: str, max_rank: str = "genus") -> str:
        """
        Get readable taxonomy string (e.g., "Firmicutes | Clostridia | Clostridiales | Lachnospiraceae | ???").
        
        Args:
            otu_id: OTU identifier
            max_rank: Deepest rank to include ("genus" by default)
            
        Returns:
            Pipe-separated taxonomy string
        """
        taxonomy_str = self.get_taxonomy(otu_id)
        if not taxonomy_str:
            return f"Unknown (OTU {otu_id})"
        
        parsed = self.parse_taxonomy(taxonomy_str)
        rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        
        if max_rank not in rank_order:
            max_rank = "genus"
        max_idx = rank_order.index(max_rank) + 1
        
        readable_parts = []
        for rank in rank_order[:max_idx]:
            readable_parts.append(parsed.get(rank, "???"))
        
        return " | ".join(readable_parts)

    def get_most_specific_taxon(self, otu_id: str) -> str:
        """
        Return species (or the deepest available rank) for plot labels.

        Rank priority: species > genus > family > order > class > phylum > kingdom.
        """
        taxonomy_str = self.get_taxonomy(otu_id)
        if not taxonomy_str:
            return f"OTU_{otu_id}"

        parsed = self.parse_taxonomy(taxonomy_str)
        rank_priority = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]

        for rank in rank_priority:
            taxon = parsed.get(rank, None)
            if taxon and taxon not in ["", "unclassified", "uncultured"]:
                return taxon

        # If no rank has a valid label, fall back to OTU ID
        return f"OTU_{otu_id}"

    def identify_cvd_mechanism(self, otu_id: str) -> Optional[Dict[str, str]]:
        """
        Identify CVD-relevant mechanism for OTU.
        
        Args:
            otu_id: OTU identifier
            
        Returns:
            Dict with mechanism info or None if no match
        """
        readable = self.get_readable_taxonomy(otu_id)
        readable_lower = readable.lower()
        
        for mechanism, info in CVD_TAXA_PATTERNS.items():
            for pattern in info["pattern"]:
                if pattern.lower() in readable_lower:
                    return {
                        "mechanism": mechanism,
                        "pattern_matched": pattern,
                        "cvd_link": info["cvd_link"],
                        "direction": info["direction"],
                        "taxonomy": readable
                    }
        
        return None
    
    def annotate_shap_features(
        self,
        feature_ids: List[str],
        shap_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Annotate OTU features with taxonomy and CVD mechanisms.
        
        Args:
            feature_ids: List of OTU IDs
            shap_values: Optional SHAP importance values
            
        Returns:
            DataFrame with columns: feature, taxonomy, mechanism, cvd_link, direction, [mean_|shap|]
        """
        rows = []
        
        for idx, fid in enumerate(feature_ids):
            taxonomy = self.get_readable_taxonomy(fid)
            mechanism_info = self.identify_cvd_mechanism(fid)
            
            row = {
                "otu_id": fid,
                "taxonomy": taxonomy,
                "mechanism": mechanism_info.get("mechanism") if mechanism_info else "uncategorized",
                "cvd_link": mechanism_info.get("cvd_link") if mechanism_info else "",
                "direction": mechanism_info.get("direction") if mechanism_info else ""
            }
            
            if shap_values is not None and idx < len(shap_values):
                row["mean_|shap|"] = shap_values[idx]
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_mechanism_summary(
        self,
        feature_df: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """
        Summarize mechanisms present in annotated features.
        
        Args:
            feature_df: Output from annotate_shap_features()
            
        Returns:
            {mechanism: [features...], ...}
        """
        summary = {}
        
        for mechanism in CVD_TAXA_PATTERNS.keys():
            mechanism_rows = feature_df[feature_df["mechanism"] == mechanism]
            if len(mechanism_rows) > 0:
                summary[mechanism] = mechanism_rows.to_dict("records")
        
        return summary


def annotate_gai_shap_report(
    gai_shap_report_path: Path,
    taxonomy_file: Path,
    output_path: Path = None
) -> Dict:
    """
    Annotate GAI SHAP report with taxonomy and CVD mechanisms.
    
    Args:
        gai_shap_report_path: Path to gai_shap_analysis_report.json
        taxonomy_file: Path to OTU taxonomy file
        output_path: Path to save annotated report (optional)
        
    Returns:
        Annotated report dictionary
    """
    import json
    
    print("Loading GAI SHAP report...")
    with open(gai_shap_report_path, 'r') as f:
        report = json.load(f)
    
    mapper = TaxonomyMapper(taxonomy_file)
    
    # Extract top features
    top_features = report.get("top_features", [])
    feature_ids = [f["feature"] for f in top_features]
    shap_values = [f["mean_|shap|"] for f in top_features]
    
    # Annotate
    annotated_df = mapper.annotate_shap_features(feature_ids, shap_values)
    mechanism_summary = mapper.create_mechanism_summary(annotated_df)
    
    # Enrich report
    report["annotated_features"] = annotated_df.to_dict("records")
    report["mechanism_summary"] = {
        mech: {
            "count": len(features),
            "top_features": features[:3],  # Top 3 by SHAP
            "description": CVD_TAXA_PATTERNS[mech]["cvd_link"],
            "direction": CVD_TAXA_PATTERNS[mech]["direction"]
        }
        for mech, features in mechanism_summary.items()
    }
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Annotated report saved to {output_path}")
    
    return report
