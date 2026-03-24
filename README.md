# curAHack 2026 Challenge 5: Microbiome-Based CVD Prediction

**Challenge leads**: Kevin Iselborn and Yuichiro Iwashita

## Introduction

Cardiovascular diseases (CVDs) kill 17.9 million people annually. Traditional risk assessment relies on blood tests, imaging, and clinical scores. But the gut microbiome — trillions of bacteria in your intestines — offers a fundamentally different window into cardiovascular health. Specific bacteria  produce metabolites like TMAO (trimethylamine N-oxide), which accelerates atherosclerosis, while others produce short-chain fatty acids (SCFAs) that reduce inflammation and protect the heart.

A 2024 paper by Bao et al. demonstrated that a Gut Age Index (GAI) — the difference between how old your gut microbiome looks vs. your actual age — can distinguish healthy from unhealthy individuals  with 66-75% balanced accuracy across 20 diseases, including cardiovascular conditions. For CVD  specifically, the paper achieved 68% balanced accuracy in the American Gut Project (AGP) cohort.

Paper: Bao, Zhiwei, et al. "Predicting host health status through an integrated machine learning framework: insights from healthy gut microbiome aging trajectory." Scientific Reports 14.1 (2024): 31143. ([Link](https://www.nature.com/articles/s41598-024-82418-3))

## Task

Develop a machine learning model that predicts individual cardiovascular risk from 16S microbiome data, leveraging microbial networks and functional patterns. To achieve this, take the GAI pipeline as your starting point and build a better CVD risk  prediction system. The paper's moderate performance (68% balanced accuracy, estimated AUC 0.58-0.70) leaves substantial room for improvement. Your target is balanced accuracy > 70% and AUC > 0.75.

What makes this a real research problem:

- The paper used raw OTU counts as features — missing complex microbial interactions and
functional pathways
- Microbiome data is compositional (relative abundances sum to a constant) and sparse (many
zeros), requiring specialized preprocessing that the paper did not fully address
- The GAI approach is indirect (predict age first, then use deviation as a disease marker) — direct
CVD classification might outperform it
- Class imbalance is severe: only 180 CVD cases vs. 1,852 healthy controls in AGP
- Cross-cohort generalization remains unsolved

## Understanding the GAI Pipeline

Before you can improve the pipeline, you need to understand exactly what it does. The GAI pipeline has three stages that transform raw microbiome data into a health status prediction.

### Stage A: Data Acquisition and Preparation

The pipeline downloads 16S rRNA sequencing data from the Qiita database using redbiom. The data comes from two sources: the Guangdong Gut Microbiome Project (GGMP, ~6,014 samples from China) and the American Gut Project (AGP, ~5,966 samples from the US). Both use Illumina 16S V4 region sequencing, processed through the Deblur pipeline for quality filtering. The raw data arrives as BIOM files (a sparse matrix of samples x Operational Taxonomic Units (OTUs)), which the prepare_data.py script converts to TSV format.

The critical preprocessing step: OTUs present in fewer than 10% of samples are removed. For GGMP, this reduces the feature space to ~942 OTUs. This prevalence filter eliminates rare, unreliable detections that add noise.

[Download AGP and GGMP datasets](https://cloud.dfki.de/owncloud/index.php/s/D8kgi7QjM8LYwyE)

### Stage B: Health Stratification

The pipeline splits participants into healthy and non-healthy groups. This is the foundation of the entire approach — the age regression model is trained only on healthy people, so it learns the normal aging trajectory. For AGP, healthy means: no cardiovascular disease, no diabetes, no cancer, no IBD, no IBS, no autoimmune disease, no kidney/liver/lung disease, no mental illness, no ASD/Alzheimer's, BMI < 24, and no antibiotic use in the past year.

### Stage C: Machine Learning Pipeline

The ML pipeline ([gai_cal.py](./gai_cal.py)) performs four operations:

#### C1. Model Training

Using PyCaret 2.3.5, the script trains multiple regression algorithms (CatBoost, LightGBM, Random Forest, Ridge, Lasso, SVR, KNN, etc.) on the healthy cohort, with OTU  abundances as features and chronological age as the target. The best model is selected by lowest MAE, then hyperparameters are tuned via random grid search. For AGP, CatBoost won; for GGMP, LightGBM won.

#### C2. Age Prediction

The finalized model predicts "gut age" for all participants (healthy and non-healthy). The paper achieved MAE of ~6.8 years, meaning predictions are off by about 6.8 years on average.

#### C3. GAI Calculation

Raw GAI = predicted gut age minus chronological age. A positive GAI means the gut looks older than expected.C4. Bias Correction: Raw GAI has a systematic bias: young people's ages are overestimated, old people's are underestimated (regression to the mean). The fix: group healthy individuals into age bins (18-20, 20-25, 25-30, ..., 75-100), compute the mean raw GAI per bin, then subtract that bin's mean from everyone in that age range. After correction, healthy people's GAI centers around zero at all ages.

**Important known bug**: The original code skips the 40-45 age bin, causing NaN values for those participants. You should fix this in your implementation.

## Getting Started

1. Clone this repository

```bash
git clone https://git.opendfki.de/yiwashita/curahack-2026-challenge-5.git
```

2. Download datasets as linked above. Both raw and processed data are provided, so you can skip the data preparation step if you want.
3. Set up a Python (3.11 recommended) environment using the following packages:

```bash
pip install numpy==2.4.3 scipy==1.17.1  pandas  matplotlib seaborn scikit-learn scikit-bio xgboost lightgbm catboost shap lime numba networkx biom-format h5py openpyxl
pip install redbiom pycaret 
```

4. Run the GAI pipeline to reproduce the paper's results:

```bash
python gai_cal.py [PATH_TO_META_FILE] [PATH_TO_OTU_FILE] [OUTPUT_DIR]
```

> [!note]
> Note we are using the latest version of PyCaret, and not the one in the original implementation, in order to use at least reasonably late versions of other packages. If you don't plan on using PyCaret you may want to skip it, in order to use newer packages.

## Contact

[firstname].[lastname]@dfki.de
