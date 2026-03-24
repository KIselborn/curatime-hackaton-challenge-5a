import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data_loader import load_agp_cvd_dataset
from src.feature_engineering import MicrobiomeFeatureEngineer
from src.gai import GutAgingIndex


def build_features_with_gai(train_otu, train_meta, target_otu, target_meta):
    fe = MicrobiomeFeatureEngineer()
    fe.fit(train_otu)

    X_train_base = fe.transform(train_otu)
    X_target_base = fe.transform(target_otu)

    gai_model = GutAgingIndex(random_state=42)
    gai_model.fit(train_otu, train_meta)

    gai_train = gai_model.transform(train_otu, train_meta)
    gai_target = gai_model.transform(target_otu, target_meta)

    X_train = X_train_base.copy()
    X_train["gai_corrected"] = gai_train

    X_target = X_target_base.copy()
    X_target["gai_corrected"] = gai_target
    return X_train, X_target, fe, gai_model


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_bal_acc = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(threshold)

    return best_threshold, float(best_bal_acc)


def estimate_threshold_cv(X_otu, meta, y, model_params, n_splits=3, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X_otu, y):
        X_train_otu = X_otu.iloc[train_idx]
        X_val_otu = X_otu.iloc[val_idx]
        meta_train = meta.iloc[train_idx]
        meta_val = meta.iloc[val_idx]
        y_train = y.iloc[train_idx]

        X_train, X_val, _, _ = build_features_with_gai(
            train_otu=X_train_otu,
            train_meta=meta_train,
            target_otu=X_val_otu,
            target_meta=meta_val,
        )

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = LGBMClassifier(**model_params)
        model.fit(X_train_res, y_train_res)
        all_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    threshold, _ = find_best_threshold(y.values, all_probs)
    return threshold


def cv_score_with_optuna_params(X_otu, meta, y, params, n_splits=3, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_bal_acc = []
    fold_auc = []

    for train_idx, val_idx in cv.split(X_otu, y):
        X_train_otu = X_otu.iloc[train_idx]
        X_val_otu = X_otu.iloc[val_idx]
        meta_train = meta.iloc[train_idx]
        meta_val = meta.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

    

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_otu, y_train)

        model = LGBMClassifier(**params)
        model.fit(X_train_res, y_train_res)

        y_prob = model.predict_proba(X_val_otu)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        threshold, bal_acc = find_best_threshold(y_val.values, y_prob)

        fold_auc.append(float(auc))
        fold_bal_acc.append(float(bal_acc))

    return float(np.mean(fold_bal_acc)), float(np.mean(fold_auc))


def tune_with_optuna(X_otu, meta, y, n_trials=20, n_splits=3, random_state=42):
    def objective(trial):
        params = {
            "class_weight": "balanced",
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "random_state": random_state,
            "verbosity": -1,
        }

        mean_bal_acc, mean_auc = cv_score_with_optuna_params(
            X_otu=X_otu,
            meta=meta,
            y=y,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
        )

        # Joint objective to improve both metrics.
        return 0.5 * mean_bal_acc + 0.5 * mean_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params.copy()
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = random_state
    best_params["verbosity"] = -1

    mean_bal_acc, mean_auc = cv_score_with_optuna_params(
        X_otu=X_otu,
        meta=meta,
        y=y,
        params=best_params,
        n_splits=n_splits,
        random_state=random_state,
    )

    return {
        "best_params": best_params,
        "cv_balanced_accuracy": mean_bal_acc,
        "cv_auc": mean_auc,
        "optuna_best_score": float(study.best_value),
    }


def run_nested_cv_report(X_otu, meta, y, outer_splits=3, inner_trials=8, inner_splits=3, random_state=42):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_bal_acc = []
    outer_auc = []

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_otu, y), start=1):
        X_outer_train = X_otu.iloc[outer_train_idx]
        y_outer_train = y.iloc[outer_train_idx]
        X_outer_test = X_otu.iloc[outer_test_idx]
        y_outer_test = y.iloc[outer_test_idx]
        meta_outer_train = meta.iloc[outer_train_idx]
        meta_outer_test = meta.iloc[outer_test_idx]

        tuned = tune_with_optuna(
            X_otu=X_outer_train,
            meta=meta_outer_train,
            y=y_outer_train,
            n_trials=inner_trials,
            n_splits=inner_splits,
            random_state=random_state + fold_idx,
        )
        params = tuned["best_params"]

        threshold = estimate_threshold_cv(
            X_otu=X_outer_train,
            meta=meta_outer_train,
            y=y_outer_train,
            model_params=params,
            n_splits=inner_splits,
            random_state=random_state + fold_idx,
        )

        

        smote = SMOTE(random_state=random_state + fold_idx)
        X_train_res, y_train_res = smote.fit_resample(X_outer_train, y_outer_train)

        model = LGBMClassifier(**params)
        model.fit(X_train_res, y_train_res)

        y_prob = model.predict_proba(X_outer_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        outer_auc.append(float(roc_auc_score(y_outer_test, y_prob)))
        outer_bal_acc.append(float(balanced_accuracy_score(y_outer_test, y_pred)))

    return {
        "nested_cv_balanced_accuracy_mean": float(np.mean(outer_bal_acc)),
        "nested_cv_balanced_accuracy_std": float(np.std(outer_bal_acc)),
        "nested_cv_auc_mean": float(np.mean(outer_auc)),
        "nested_cv_auc_std": float(np.std(outer_auc)),
        "outer_splits": outer_splits,
        "inner_trials": inner_trials,
        "inner_splits": inner_splits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--nested-outer", type=int, default=3)
    parser.add_argument("--nested-inner-trials", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AGP CVD dataset...")
    meta, otu, y = load_agp_cvd_dataset(str(data_dir))
    print(f"Samples: {len(y)}, OTU features: {otu.shape[1]}")
    print("Class distribution:")
    print(y.value_counts())

    meta, otu,_,_  = build_features_with_gai(
        train_otu=otu,
        train_meta=meta,
        target_otu=otu,
        target_meta=meta,
        )

    X_train_otu, X_test_otu, meta_train, meta_test, y_train, y_test = train_test_split(
        otu,
        meta,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Running Optuna tuning...")
    tuning = tune_with_optuna(
        X_otu=X_train_otu,
        meta=meta_train,
        y=y_train,
        n_trials=args.optuna_trials,
        n_splits=args.cv_splits,
        random_state=42,
    )
    best_params = tuning["best_params"]

    print("Estimating optimal threshold from CV on training split...")
    threshold = estimate_threshold_cv(
        X_otu=X_train_otu,
        meta=meta_train,
        y=y_train,
        model_params=best_params,
        n_splits=args.cv_splits,
        random_state=42,
    )

    print("Training final model on holdout split...")
    

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_otu, y_train)

    model = LGBMClassifier(**best_params)
    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test_otu)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    test_auc = float(roc_auc_score(y_test, y_prob))
    test_bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    metrics = {
        "holdout_balanced_accuracy": test_bal_acc,
        "holdout_auc": test_auc,
        "threshold": float(threshold),
        "n_samples": int(len(y)),
        "n_engineered_features": int(X_train_otu.shape[1]),
        "optuna": tuning,
    }

    if args.nested:
        print("Running nested CV report...")
        nested = run_nested_cv_report(
            X_otu=otu,
            meta=meta,
            y=y,
            outer_splits=args.nested_outer,
            inner_trials=args.nested_inner_trials,
            inner_splits=args.cv_splits,
            random_state=42,
        )
        metrics["nested_cv"] = nested

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, output_dir / "lgbm_cvd_model.joblib")
    #joblib.dump(fe, output_dir / "feature_engineer.joblib")
    #joblib.dump(gai_model, output_dir / "gai_model.joblib")

    # ─── SHAP Analysis for Model Interpretability ───────────────────────────────
    print("\n=== SHAP-Based Interpretability Analysis ===")
    taxonomy_file = Path(__file__).parent / "97_otu_taxonomy.txt"
    
    shap_results = run_shap_analysis(
        model=model,
        X_test=X_test,
        output_dir=output_dir / "shap_analysis",
        check_additivity=False,
        taxonomy_file=taxonomy_file if taxonomy_file.exists() else None
    )
    
    # Run GAI-level SHAP analysis (interpreting drivers of gai_corrected)
    gai_shap_results = run_gai_shap_analysis(
        gai_model=gai_model,
        otu_df=X_test_otu,  # Use test OTU data for SHAP explanations
        output_dir=output_dir / "gai_shap_analysis",
        check_additivity=False,
        n_top=30,
        taxonomy_file=taxonomy_file if taxonomy_file.exists() else None
    )

    # Annotate GAI SHAP report with taxonomy and CVD mechanisms
    print("\n=== Annotating SHAP Features with Taxonomy ===")
    taxonomy_file = Path(__file__).parent / "97_otu_taxonomy.txt"
    if taxonomy_file.exists():
        try:
            gai_report_path = output_dir / "gai_shap_analysis" / "gai_shap_analysis_report.json"
            gai_annotated_path = output_dir / "gai_shap_analysis" / "gai_shap_analysis_annotated.json"
            annotate_gai_shap_report(
                gai_shap_report_path=gai_report_path,
                taxonomy_file=taxonomy_file,
                output_path=gai_annotated_path
            )
            print(f"✓ Annotated GAI SHAP report saved to: {gai_annotated_path}")
            
            # Generate biological summary
            print("\n=== Generating Biological Summary ===")
            from create_biological_summary import create_biological_summary
            summary_result = create_biological_summary(
                shap_report_path=gai_annotated_path,
                output_path=output_dir / "gai_shap_analysis" / "biological_summary.html"
            )
            print(f"✓ Biological summary HTML: {summary_result['html_report']}")
            
        except Exception as e:
            print(f"⚠ Taxonomy annotation or summary generation failed: {str(e)}")
    else:
        print(f"⚠ Taxonomy file not found at {taxonomy_file}. Skipping annotation and summary.")

    # Save SHAP analysis summary to metrics
    metrics["shap_analysis"] = {
        "status": "completed",
        "top_5_features": shap_results["importance_df"].head(5)["feature"].tolist(),
        "biomarkers_detected": list(shap_results["biomarker_map"].keys()),
        "report_path": "shap_analysis/shap_analysis_report.json",
        "plots": {
            "summary": "shap_analysis/shap_summary.png",
            "dependence": "shap_analysis/shap_dependence/"
        }
    }
    metrics["gai_shap_analysis"] = {
        "status": "completed",
        "top_5_features": gai_shap_results["importance_df"].head(5)["feature"].tolist(),
        "biomarkers_detected": list(gai_shap_results["biomarker_map"].keys()),
        "report_path": "gai_shap_analysis/gai_shap_analysis_report.json",
        "plots": {
            "summary": "gai_shap_analysis/gai_shap_summary.png",
            "dependence": "gai_shap_analysis/gai_shap_dependence/"
        }
    }

    joblib.dump(model, output_dir / "lgbm_cvd_model.joblib")
    joblib.dump(fe, output_dir / "feature_engineer.joblib")
    joblib.dump(gai_model, output_dir / "gai_model.joblib")
    
    # Save updated metrics with SHAP results
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    
    print("\n=== SHAP Analysis Summary ===")
    print("Top 5 Predictive Features:")
    for idx, (_, row) in enumerate(shap_results["importance_df"].head(5).iterrows(), 1):
        print(f"  {idx}. {row['feature']:<40} (|SHAP|: {row['mean_|shap|']:.4f})")
    
    print("\nBiomarker Mechanisms Identified:")
    for mechanism, features in shap_results["biomarker_map"].items():
        if mechanism != "unclassified" and features:
            print(f"  • {mechanism}: {len(features)} feature(s) detected")
    
    print("\nBiological Insights:")
    for insight in shap_results["insights"].get("publication_value", [])[:3]:
        print(f"  • {insight}")
    
    print(f"\nInterpretability Report: {output_dir}/shap_analysis/shap_analysis_report.json")
    print(f"Summary Plot: {output_dir}/shap_analysis/shap_summary.png")
    
    print("\n=== Holdout Evaluation ===")
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"AUC-ROC:           {test_auc:.4f}")
    print(f"Threshold:         {threshold:.3f}")
    if "nested_cv" in metrics:
        nested = metrics["nested_cv"]
        print("\n=== Nested CV Summary ===")
        print(f"Balanced Accuracy: {nested['nested_cv_balanced_accuracy_mean']:.4f} ± {nested['nested_cv_balanced_accuracy_std']:.4f}")
        print(f"AUC-ROC:           {nested['nested_cv_auc_mean']:.4f} ± {nested['nested_cv_auc_std']:.4f}")


if __name__ == "__main__":
    main()
