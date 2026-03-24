"""
SHAP-based Model Interpretability Analysis with Biological Insights
Connects SHAP-identified features to known CVD mechanisms and microbiome biology.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
# Set random seeds for deterministic SHAP values
np.random.seed(42)
try:
    import random
    random.seed(42)
except:
    pass

# ─── Known CVD-Related Microbiome Biomarkers ────────────────────────────────────
# Maps OTU patterns and features to known biological mechanisms

CVD_BIOMARKER_MECHANISMS = {
    "tmao_pathway": {
        "description": "TMAO (trimethylamine N-oxide) production pathway",
        "mechanisms": [
            "Elevated Firmicutes (especially Clostridium, Ruminococcus)",
            "Reduced Bacteroidetes (decreased TMAO conversion capacity)",
            "Elevated TMAO-producer taxa (Prevotella, Alistipes)",
            "Altered Firmicutes/Bacteroidetes ratio"
        ],
        "cvd_link": "TMAO is a potent atherogenic metabolite elevated in CVD patients",
        "taxa_patterns": [
            "Clostridium", "Ruminococcus", "Prevotella", "Alistipes",
            "Firmicutes", "Bacteroidetes"
        ]
    },
    "akkermansia": {
        "description": "Akkermansia muciniphila - barrier function and inflammation",
        "mechanisms": [
            "Reduced Akkermansia muciniphila abundance",
            "Compromised intestinal barrier",
            "Increased lipopolysaccharide (LPS) translocation",
            "Elevated endotoxemia"
        ],
        "cvd_link": "Low Akkermansia associated with increased CVD risk via enhanced endotoxemia",
        "taxa_patterns": ["Akkermansia", "Verrucomicrobia"]
    },
    "butyrate_producers": {
        "description": "Short-chain fatty acid (SCFA) producers - barrier and immune",
        "mechanisms": [
            "Faecalibacterium prausnitzii (F. prausnitzii) - primary butyrate producer",
            "Roseburia species - secondary SCFA producers",
            "Reduced SCFA production capacity",
            "Decreased intestinal tight junction integrity"
        ],
        "cvd_link": "Low butyrate producers associated with increased intestinal permeability and CVD",
        "taxa_patterns": ["Faecalibacterium", "Roseburia", "Butyrate_producers"]
    },
    "diversity_evenness": {
        "description": "Alpha diversity and community evenness",
        "mechanisms": [
            "Reduced microbial diversity",
            "Dysbiosis phenotype",
            "Altered metabolic capacity",
            "Loss of protective commensals"
        ],
        "cvd_link": "Low diversity associated with metabolic dysfunction and CVD progression",
        "taxa_patterns": ["alpha_diversity", "shannon", "chao1", "evenness"]
    },
    "lps_producers": {
        "description": "Gram-negative bacteria and LPS production",
        "mechanisms": [
            "Proteobacteria abundance",
            "Inflammation-inducing LPS from Gram-negative taxa",
            "Enhanced TLR4 signaling",
            "Systemic inflammation"
        ],
        "cvd_link": "Elevated LPS-producing taxa promote systemic inflammation in CVD",
        "taxa_patterns": ["Proteobacteria", "Escherichia", "Klebsiella"]
    },
    "secondary_bile_salt_metabolizers": {
        "description": "Secondary bile acid (BA) metabolism",
        "mechanisms": [
            "Bile acid-metabolizing bacteria",
            "FXR and TGR5 ligand production",
            "Altered BA circulation",
            "Disrupted metabolism"
        ],
        "cvd_link": "Dysbiosis in BA metabolism affects farnesoid X receptor signaling and CVD risk",
        "taxa_patterns": ["Clostridium", "Bacteroides", "Parabacteroides"]
    }
}


class SHAPAnalyzer:
    """
    Performs SHAP-based model interpretation with biological insights for CVD prediction.
    """
    
    def __init__(self, model: LGBMClassifier, feature_names: List[str], 
                 feature_labels: Dict[str, str] = None):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained LGBMClassifier
            feature_names: List of feature names (e.g., OTU IDs or engineered features)
            feature_labels: Optional dict mapping feature names to readable labels
                           (e.g., {"1096610": "Firmicutes | Clostridia | ..."})
        """
        self.model = model
        self.feature_names = np.array(feature_names)
        self.feature_labels = feature_labels or {}
        self.explainer = None
        self.shap_values = None
        self.shap_base_value = None
        self.feature_importance_summary = None
    
    def set_feature_labels(self, feature_labels: Dict[str, str]):
        """Update feature label mapping for plots."""
        self.feature_labels = feature_labels
        
    def explain(self, X: pd.DataFrame, X_train: pd.DataFrame = None, check_additivity: bool = False) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X: Input features (samples x features) - test data for explanations
            X_train: DEPRECATED - ignored. TreeExplainer now uses only test data.
            check_additivity: Whether to check SHAP additivity (slower but validates correctness)
            
        Returns:
            Dictionary with SHAP values and base value
        """
        # Ensure deterministic behavior
        np.random.seed(42)
        
        print("  Initializing TreeExplainer with test data only...")
        # Use test data itself as the background for TreeExplainer
        # This ensures each sample is explained based on its actual distribution
        self.explainer = shap.TreeExplainer(self.model, data=X)
        
        print(f"  Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        # For binary classification, shap_values is a list [neg_class, pos_class]
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Focus on positive class
        
        self.shap_base_value = self.explainer.expected_value
        if isinstance(self.shap_base_value, (list, np.ndarray)):
            self.shap_base_value = self.shap_base_value[1]
        
        return {
            "shap_values": self.shap_values,
            "base_value": self.shap_base_value,
            "features": self.feature_names
        }
    
    def get_feature_importance(self, n_top: int = 20, 
                               abs_importance: bool = True) -> pd.DataFrame:
        """
        Rank features by average absolute SHAP value.
        
        Args:
            n_top: Number of top features to return
            abs_importance: Use absolute SHAP values (default: True)
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        if abs_importance:
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        else:
            mean_abs_shap = self.shap_values.mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_|shap|": mean_abs_shap,
        }).sort_values("mean_|shap|", ascending=False)
        
        self.feature_importance_summary = importance_df.head(n_top).copy()
        return importance_df.head(n_top)
    
    def identify_biomarker_features(self, 
                                   importance_df: pd.DataFrame = None,
                                   n_top: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map top SHAP features to known CVD microbiome biomarkers.
        
        Args:
            importance_df: Output from get_feature_importance() 
            n_top: Number of top features to analyze
            
        Returns:
            Dictionary mapping biomarker mechanisms to identified features
        """
        if importance_df is None:
            importance_df = self.get_feature_importance(n_top=n_top)
        
        top_features = importance_df["feature"].values
        biomarker_mapping = {mech: [] for mech in CVD_BIOMARKER_MECHANISMS.keys()}
        unmapped = []
        
        for feature in top_features:
            feature_lower = feature.lower()
            mapped = False
            
            for biomarker, info in CVD_BIOMARKER_MECHANISMS.items():
                # Check if feature matches any known taxa or pattern
                for pattern in info["taxa_patterns"]:
                    if pattern.lower() in feature_lower:
                        idx = importance_df[importance_df["feature"] == feature].index[0]
                        importance_val = importance_df.loc[idx, "mean_|shap|"]
                        
                        biomarker_mapping[biomarker].append({
                            "feature": feature,
                            "mean_|shap|": float(importance_val),
                            "mechanism": biomarker,
                            "description": info["description"]
                        })
                        mapped = True
                        break
            
            if not mapped:
                unmapped.append(feature)
        
        # Remove empty biomarkers and add unmapped features
        biomarker_mapping = {k: v for k, v in biomarker_mapping.items() if v}
        if unmapped:
            biomarker_mapping["unclassified"] = [
                {"feature": f, "mean_|shap|": 0.0, "mechanism": "unclassified"} 
                for f in unmapped
            ]
        
        return biomarker_mapping
    
    def generate_biological_insights(self, 
                                    importance_df: pd.DataFrame = None,
                                    shap_values_array: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate biological interpretation of top features.
        
        Args:
            importance_df: Output from get_feature_importance()
            shap_values_array: The SHAP values array (for summary)
            
        Returns:
            Dictionary with biological insights and interpretations
        """
        if importance_df is None:
            importance_df = self.get_feature_importance()
        
        biomarker_map = self.identify_biomarker_features(importance_df)
        
        insights = {
            "summary": "SHAP-identified features mapped to known CVD microbiome mechanisms",
            "top_biomarkers": {},
            "mechanistic_insights": {},
            "publication_value": []
        }
        
        # Organize by biomarker mechanism
        for biomarker, features in sorted(biomarker_map.items(), 
                                         key=lambda x: sum(f.get("mean_|shap|", 0) 
                                                          for f in x[1]), 
                                         reverse=True):
            if features and biomarker != "unclassified":
                mech_info = CVD_BIOMARKER_MECHANISMS.get(biomarker, {})
                insights["top_biomarkers"][biomarker] = {
                    "description": mech_info.get("description"),
                    "cvd_link": mech_info.get("cvd_link"),
                    "top_features": features[:3]
                }
                insights["publication_value"].append(
                    f"Model identified {biomarker}: {mech_info.get('cvd_link')}"
                )
        
        return insights
    
    def create_summary_plot(self, X: pd.DataFrame, output_path: Path = None, 
                           max_display: int = 15) -> str:
        """
        Create SHAP summary plot (bar plot of feature importance).
        
        Args:
            X: Input features
            output_path: Path to save plot (if None, creates temporary file)
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        if output_path is None:
            output_path = Path("/tmp/shap_summary.png")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use readable labels if available
        display_names = self.feature_names.copy()
        if self.feature_labels:
            display_names = np.array([
                self.feature_labels.get(name, name)[:80]  # Truncate to 80 chars
                for name in self.feature_names
            ])
        
        plt.figure(figsize=(12, max_display * 0.4))  # Increased width and height
        shap.summary_plot(self.shap_values, X, feature_names=display_names,
                         plot_type="bar", max_display=max_display, show=False)
        plt.suptitle("SHAP Feature Importance: Average Impact on Model Output", fontsize=12, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return str(output_path)
    
    def create_dependence_plots(self, X: pd.DataFrame, output_dir: Path = None, 
                               n_features: int = 5) -> List[str]:
        """
        Create SHAP dependence plots for top features.
        
        Args:
            X: Input features
            output_dir: Directory to save plots
            n_features: Number of top features to plot
            
        Returns:
            List of paths to saved plots
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        if output_dir is None:
            output_dir = Path("/tmp/shap_dependence")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use readable labels if available
        display_names = self.feature_names.copy()
        if self.feature_labels:
            display_names = np.array([
                self.feature_labels.get(name, name)[:80]
                for name in self.feature_names
            ])
        
        importance_df = self.get_feature_importance(n_top=n_features)
        saved_plots = []
        
        for idx, (_, row) in enumerate(importance_df.iterrows()):
            feature = row["feature"]
            feature_idx = np.where(self.feature_names == feature)[0][0]
            
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature_idx, self.shap_values, X, 
                                feature_names=display_names, show=False)
            
            plot_path = output_dir / f"shap_dependence_{idx:02d}_{feature}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(str(plot_path))
        
        return saved_plots
    
    def export_interpretable_report(self, output_path: Path) -> None:
        """
        Export comprehensive SHAP analysis report as JSON.
        
        Args:
            output_path: Path to save JSON report
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        importance_df = self.get_feature_importance(n_top=30)
        biomarker_map = self.identify_biomarker_features(importance_df, n_top=30)
        insights = self.generate_biological_insights(importance_df)
        
        report = {
            "analysis_type": "SHAP TreeExplainer",
            "model_type": "LGBMClassifier",
            "base_value": float(self.shap_base_value),
            "n_samples_explained": int(len(self.shap_values)),
            "n_features": int(len(self.feature_names)),
            "top_features": importance_df.head(20).to_dict("records"),
            "biomarker_mechanisms": {
                mech: [
                    {k: (v if k != "mean_|shap|" else float(v)) for k, v in f.items()}
                    for f in features
                ]
                for mech, features in biomarker_map.items()
            },
            "biological_insights": insights,
            "mechanisms_detected": list(CVD_BIOMARKER_MECHANISMS.keys()),
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ SHAP analysis report saved to {output_path}")


def run_gai_shap_analysis(
    gai_model,
    otu_df: pd.DataFrame,
    output_dir: Path,
    otu_train: pd.DataFrame = None,
    n_top: int = 30,
    check_additivity: bool = False,
    taxonomy_file: Path = None
) -> Dict[str, Any]:
    """
    Run SHAP analysis for GutAgingIndex internal regressor, interpretation of 'gai_corrected'.

    Args:
        gai_model: Fitted GutAgingIndex instance (must have .regressor)
        otu_df: OTU feature DataFrame used by GAI model (test data for both explanations and background)
        output_dir: Directory to save GAI SHAP outputs
        otu_train: DEPRECATED - ignored. TreeExplainer now uses only test data.
        n_top: Number of top features to extract
        check_additivity: Whether to validate additivity of SHAP values
        taxonomy_file: Optional path to taxonomy file for readable feature labels

    Returns:
        Dictionary with analysis results for GAI
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(gai_model, "regressor"):
        raise ValueError("gai_model must have .regressor attribute (lightgbm regressor)")

    print("\n=== GAI SHAP Analysis (gai_corrected drivers) ===")
    model = gai_model.regressor
    analyzer = SHAPAnalyzer(model, list(otu_df.columns))
    
    # Load taxonomy labels if provided
    if taxonomy_file and Path(taxonomy_file).exists():
        try:
            from src.taxonomy_mapper import TaxonomyMapper
            print("  Loading taxonomy mappings for better feature labels...")
            mapper = TaxonomyMapper(taxonomy_file)
            feature_labels = {}
            for otu_id in otu_df.columns:
                label = mapper.get_most_specific_taxon(otu_id)
                if label:
                    feature_labels[otu_id] = label
            if feature_labels:
                analyzer.set_feature_labels(feature_labels)
                print(f"  ✓ Loaded {len(feature_labels)} taxonomy labels")
        except Exception as e:
            print(f"  ⚠ Failed to load taxonomy: {str(e)}")

    print("Step 1: Computing SHAP values for GAI regressor...")
    analyzer.explain(otu_df, check_additivity=check_additivity)

    print("Step 2: Ranking features by importance for GAI...")
    importance_df = analyzer.get_feature_importance(n_top=n_top)
    print("Top 10 GAI drivers by SHAP importance:")
    print(importance_df.head(10).to_string())

    print("Step 3: Mapping GAI features to known CVD biomarkers...")
    biomarker_map = analyzer.identify_biomarker_features(importance_df, n_top=n_top)
    for mechanism, features in biomarker_map.items():
        if mechanism != "unclassified" and features:
            print(f"  ✓ GAI {mechanism}: {len(features)} feature(s)")

    print("Step 4: Creating GAI visualizations...")
    summary_plot = analyzer.create_summary_plot(otu_df, output_path=output_dir / "gai_shap_summary.png", max_display=20)
    dependence_plots = analyzer.create_dependence_plots(otu_df, output_dir=output_dir / "gai_shap_dependence", n_features=5)

    print("Step 5: Exporting GAI report...")
    analyzer.export_interpretable_report(output_dir / "gai_shap_analysis_report.json")

    return {
        "analyzer": analyzer,
        "importance_df": importance_df,
        "biomarker_map": biomarker_map,
        "plots": {
            "summary": summary_plot,
            "dependence": dependence_plots,
        },
    }


def run_shap_analysis(model: LGBMClassifier, 
                     X_test: pd.DataFrame,
                     output_dir: Path,
                     X_train: pd.DataFrame = None,
                     check_additivity: bool = False,
                     taxonomy_file: Path = None) -> Dict[str, Any]:
    """
    Convenience function to run complete SHAP analysis pipeline.
    
    Args:
        model: Trained LGBMClassifier
        X_test: Test set features (used for both explanations and background)
        output_dir: Directory to save analysis outputs
        X_train: DEPRECATED - ignored. TreeExplainer now uses only test data.
        check_additivity: Whether to validate SHAP additivity
        taxonomy_file: Optional path to taxonomy file for readable feature labels
        
    Returns:
        Dictionary with analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SHAP-Based Model Interpretability Analysis ===")
    analyzer = SHAPAnalyzer(model, list(X_test.columns))
    
    # Load taxonomy labels if provided
    if taxonomy_file and Path(taxonomy_file).exists():
        try:
            from src.taxonomy_mapper import TaxonomyMapper
            print("  Loading taxonomy mappings for better feature labels...")
            mapper = TaxonomyMapper(taxonomy_file)
            feature_labels = {}
            for otu_id in X_test.columns:
                if otu_id != "gai_corrected":  # Skip non-OTU features
                    label = mapper.get_most_specific_taxon(otu_id)
                    if label:
                        feature_labels[otu_id] = label
            if feature_labels:
                analyzer.set_feature_labels(feature_labels)
                print(f"  ✓ Loaded {len(feature_labels)} taxonomy labels")
        except Exception as e:
            print(f"  ⚠ Failed to load taxonomy: {str(e)}")

    print("Step 1: Computing SHAP values...")
    analyzer.explain(X_test, check_additivity=check_additivity)
    
    print("Step 2: Ranking features by importance...")
    importance_df = analyzer.get_feature_importance(n_top=30)
    print("Top 10 features by SHAP importance:")
    print(importance_df.head(10).to_string())
    
    print("\nStep 3: Mapping to known CVD biomarkers...")
    biomarker_map = analyzer.identify_biomarker_features(importance_df, n_top=30)
    for mechanism in biomarker_map:
        if mechanism != "unclassified" and biomarker_map[mechanism]:
            print(f"  ✓ {mechanism}: {len(biomarker_map[mechanism])} feature(s)")
    
    print("\nStep 4: Generating biological insights...")
    insights = analyzer.generate_biological_insights(importance_df)
    for insight in insights.get("publication_value", []):
        print(f"  • {insight}")
    
    print("\nStep 5: Creating visualizations...")
    summary_plot = analyzer.create_summary_plot(X_test, 
                                               output_path=output_dir / "shap_summary.png")
    print(f"  ✓ Summary plot: {Path(summary_plot).name}")
    
    dependence_plots = analyzer.create_dependence_plots(X_test, 
                                                       output_dir=output_dir / "shap_dependence",
                                                       n_features=5)
    print(f"  ✓ Dependence plots: {len(dependence_plots)} created")
    
    print("\nStep 6: Exporting comprehensive report...")
    analyzer.export_interpretable_report(output_dir / "shap_analysis_report.json")
    
    return {
        "analyzer": analyzer,
        "importance_df": importance_df,
        "biomarker_map": biomarker_map,
        "insights": insights,
        "plots": {
            "summary": summary_plot,
            "dependence": dependence_plots
        }
    }
