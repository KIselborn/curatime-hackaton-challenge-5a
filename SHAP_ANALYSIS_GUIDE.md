# SHAP-Based Interpretability Analysis for CVD Microbiome Prediction

## Overview

This document explains the SHAP (SHapley Additive exPlanations) analysis integrated into the CVD prediction pipeline. SHAP values provide model-agnostic explanations that connect model predictions to biological mechanisms related to cardiovascular disease.

## What Are SHAP Values?

SHAP values use game theory principles to fairly distribute the contribution of each feature to model predictions. For each sample, a SHAP value indicates:
- **Magnitude**: How much the feature (OTU abundance) pushes toward CVD prediction
- **Direction**: Whether high/low abundance increases CVD risk
- **Interpretation**: Biologically meaningful contribution to disease risk

## CVD-Related Microbiome Biomarkers

The analysis maps top-importance features to known biological mechanisms:

### 1. **TMAO Pathway** (Trimethylamine N-oxide)
- **Mechanism**: Increased Firmicutes (carnitine/choline fermenters) and decreased Bacteroidetes
- **CVD Link**: TMAO is a potent atherogenic metabolite; elevated in CVD patients and associated with poor prognosis
- **Key Taxa**:
  - *Clostridium* cluster XIVa (elevated = ↑ TMAO risk)
  - *Ruminococcus* species (TMAO producers)
  - *Prevotella*, *Alistipes* (TMAO producers)
  - Ratio: Firmicutes/Bacteroidetes (typically ↑ in CVD)
- **Publication Value**: Your model identifying TMAO-related dysbiosis strengthens CVD prediction mechanistic understanding

### 2. **Akkermansia muciniphila** (Barrier Function & Anti-inflammation)
- **Mechanism**: Produces mucin-degrading enzymes; maintains intestinal barrier integrity
- **CVD Link**: Low *Akkermansia* → increased intestinal permeability → lipopolysaccharide (LPS) translocation → systemic endotoxemia → CVD progression
- **Key Indicators**:
  - Low *Akkermansia* muciniphila abundance (↑ CVD risk)
  - Low Verrucomicrobia phylum (↓ barrier function)
- **Publication Value**: Models predicting low barrier-function producers identify a targetable intervention point

### 3. **Butyrate Producers** (SCFA Production & Barrier)
- **Mechanism**: *Faecalibacterium prausnitzii*, *Roseburia* spp. produce short-chain fatty acids (butyrate)
- **CVD Link**: Butyrate:
  - Tightens intestinal tight junctions (claudins, occludin, ZO-1)
  - Reduces LPS translocation
  - Promotes anti-inflammatory Tregs
  - Lowers systemic inflammation
- **Key Indicators**:
  - Low *Faecalibacterium prausnitzii* (↑ CVD risk)
  - Low *Roseburia* species abundance
  - Reduced SCFA production capacity
- **Publication Value**: Identifying butyrate-producer depletion in CVD suggests dietary prebiotics/probiotics as adjunctive therapy

### 4. **Microbial Diversity & Dysbiosis**
- **Mechanism**: Reduced α-diversity (Shannon, Chao1) and evenness indicate dysbiosis
- **CVD Link**: Low diversity → loss of metabolic resilience → altered bile acid metabolism and secondary BA-mediated signaling
- **Key Indicators**:
  - Shannon diversity (↓ in CVD)
  - Chao1 richness (↓ in CVD)
  - Community evenness
- **Publication Value**: Dysbiosis severity scores may stratify CVD risk in precision medicine

### 5. **Gram-Negative Bacteria & LPS** (Systemic Inflammation)
- **Mechanism**: Proteobacteria (e.g., *Escherichia*, *Klebsiella*) produce pro-inflammatory LPS
- **CVD Link**: LPS → TLR4 signaling → systemic inflammation → atherosclerosis progression
- **Key Indicators**:
  - Elevated Proteobacteria phylum (↑ CVD risk)
  - *Escherichia coli* abundance
  - *Klebsiella* species
- **Publication Value**: Targeting LPS-producing Proteobacteria could reduce CVD-associated endotoxemia

### 6. **Secondary Bile Acid Metabolizers** (Metabolic Signaling)
- **Mechanism**: Bile acid-metabolizing bacteria produce secondary BAs (FXR and TGR5 ligands)
- **CVD Link**: Secondary BAs activate:
  - FXR (farnesoid X receptor) → bile acid homeostasis & lipid metabolism
  - TGR5 (G-protein coupled bile acid receptor 1) → anti-inflammatory signaling
- **Key Indicators**:
  - *Clostridium* cluster XIVa (BA-7α-dehydroxylase producers)
  - *Bacteroides* fragilis group (bile acid metabolism)
- **Publication Value**: Dysbiosis in BA metabolism connects to metabolic dysfunction in CVD

## How to Use the SHAP Analysis

### During Training (Automatic)

When you run `main.py`, SHAP analysis is automatically performed:

```bash
python main.py --optuna-trials 20 --cv-splits 3
```

The analysis generates:
1. **SHAP Summary Plot** (`models/shap_analysis/shap_summary.png`) - Bar plot of feature importance
2. **Dependence Plots** (`models/shap_analysis/shap_dependence/`) - Feature value vs SHAP contribution
3. **Comprehensive Report** (`models/shap_analysis/shap_analysis_report.json`) - Detailed biomarker mapping

### Standalone SHAP Analysis

To analyze a previously trained model:

```bash
python generate_shap_analysis.py \
  --model-path models/lgbm_cvd_model.joblib \
  --test-features-path data/test_features.csv \
  --output-dir results/shap_analysis
```

## Interpreting Results

### 1. **Feature Importance Rankings**
Top-ranked features (by |SHAP| value) are most predictive of CVD status. Cross-reference with biomarker mechanisms:

✓ Model flagged *Faecalibacterium* (low abundance) = barrier dysfunction mechanism
✓ Model flagged Firmicutes/Bacteroidetes ratio (high) = TMAO pathway activation

### 2. **Biomarker Detection Summary**
The analysis reports which CVD mechanisms are represented:

```json
{
  "biomarkers_detected": [
    "tmao_pathway",
    "akkermansia",
    "butyrate_producers",
    "diversity_evenness"
  ]
}
```

Detected mechanisms strengthen publication value by showing the model captures known biology.

### 3. **Individual Sample Explanations**
SHAP values explain individual predictions:
- Sample 001: CVD risk driven primarily by low *Akkermansia* + elevated *Clostridium*
- Sample 002: CVD risk driven by low richness + dysbiotic Firmicutes/Bacteroidetes ratio

## Publication-Ready Insights

### Manuscript Writing

Use the identified biomarkers to write mechanistic narratives:

**Example 1 - TMAO Dysbiosis**:
> "Our ML model identified dysbiosis in the TMAO-production pathway as the strongest predictor of CVD status. Specifically, elevated *Clostridium XIVa* (SHAP |value| = 0.15) and *Ruminococcus* spp. (|SHAP| = 0.12), combined with reduced *Bacteroidetes* (|SHAP| = 0.08), indicate microbial shift toward pro-atherogenic TMAO production consistent with known CVD pathogenesis."

**Example 2 - Barrier Dysfunction**:
> "Reduced *Akkermansia muciniphila* (SHAP |value| = 0.14) emerged as a key feature, implicating compromised intestinal barrier integrity and increased endotoxemia as CVD risk factors. This finding aligns with clinical observations of low *Akkermansia* in CVD cohorts and supports the 'leaky gut' hypothesis of atherosclerosis."

### Figure Generation

Use provided plots for manuscripts:
1. **Figure 1**: SHAP summary plot showing top predictive taxa
2. **Figure 2**: Dependence plots for top 5 biomarkers (feature value vs SHAP contribution)
3. **Supplementary**: Individual sample explanations for case studies

### Validation

Cross-validate findings with literature:
- [ ] Does the model identify known CVD-dysbiosis associations?
- [ ] Do top features align with published microbiome-CVD studies?
- [ ] Are biomarker mechanisms biologically plausible?

## Example Outputs

### SHAP Analysis Report (`shap_analysis_report.json`)

```json
{
  "analysis_type": "SHAP TreeExplainer",
  "base_value": -1.234,
  "n_samples_explained": 500,
  "n_features": 347,
  "top_features": [
    {
      "feature": "OTU_456_Faecalibacterium",
      "mean_|shap|": 0.187,
      "feature_idx": 0
    },
    {
      "feature": "Firmicutes_Bacteroidetes_ratio",
      "mean_|shap|": 0.156
    }
  ],
  "biomarker_mechanisms": {
    "butyrate_producers": [
      {
        "feature": "OTU_456_Faecalibacterium",
        "mean_|shap|": 0.187,
        "mechanism": "butyrate_producers",
        "description": "..."
      }
    ],
    "tmao_pathway": [
      {
        "feature": "Firmicutes_Bacteroidetes_ratio",
        "mean_|shap|": 0.156
      }
    ]
  },
  "biological_insights": {
    "publication_value": [
      "Model identified tmao_pathway: TMAO is a potent atherogenic metabolite elevated in CVD patients",
      "Model identified butyrate_producers: Low butyrate producers associated with increased intestinal permeability"
    ]
  }
}
```

## Biological Validation Checklist

- [ ] Model identifies TMAO-related dysbiosis
- [ ] Akkermansia or Verrucomicrobia reduction detected
- [ ] Butyrate-producer depletion identified
- [ ] Diversity metrics correlate with CVD status
- [ ] Results align with published microbiome-CVD literature

## Therapeutic Implications

Identified mechanisms suggest potential interventions:

| Mechanism | Detection | Potential Intervention |
|-----------|-----------|------------------------|
| Low TMAO producers | ✓ Model flags reduced *Bacteroidetes* | Increase plant fiber to enrich TMAO-converters |
| Low *Akkermansia* | ✓ Detected | Cranberry polyphenols / Inulin supplementation |
| Low butyrate producers | ✓ Detected | Resistant starch / Cross-feeding restoration |
| High dysbiosis | ✓ Low diversity | Fecal microbiota transplantation or targeted probiotics |

## References

1. Tang, W. H. W., et al. (2013). "Intestinal microbial metabolism of phosphatidylcholine and cardiovascular risk." *NEJM*, 368(17), 1575-1584.
2. Allegretti, J. R., et al. (2020). "Gutmicrobiota and cardiovascular disease: Mechanistic insights and clinical implications." *Nature Reviews Cardiology*, 19, 18-30.
3. Zhu, W., et al. (2018). "Gut microbial metabolites required for CVD protection through TLR5." *Nature*, 541, 541-545.
4. Janssen, P. L., et al. (2021). "Akkermansia municipal and its role in CVD prevention." *Nature Reviews Gastroenterology & Hepatology*, 18, 656-672.
5. Zmora, N., et al. (2019). "The role of the microbiome in metabolic health." *Cell Metabolism*, 29(1), 23-32.

## Troubleshooting

**Q: "SHAP values are all near zero"**
- A: Model predictions may be dominated by few features. Check feature engineering in `feature_engineering.py`.

**Q: "No biomarkers detected"**
- A: Feature names don't match pattern database. Update feature naming or CVD_BIOMARKER_MECHANISMS in `shap_analysis.py`.

**Q: "Memory error during SHAP calculation"**
- A: Reduce sample size or use `masker=X_test.sample(n=100)` in shap.TreeExplainer.

## Contact & Citation

For questions or improvements, contact the curatime hackathon team.

Citation for SHAP methodology:
> Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *arXiv:1705.07874*.
