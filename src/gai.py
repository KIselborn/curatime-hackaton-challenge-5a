import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


AGE_BINS: list[tuple[int, int]] = [
    (18, 20),
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, 60),
    (60, 65),
    (65, 70),
    (70, 75),
    (75, 100),
]


class GutAgingIndex:
    """
    Fit age regressor on healthy samples only, then compute bias-corrected GAI.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regressor = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=random_state,
            verbosity=-1,
        )
        self.bin_adjustments_: dict[tuple[int, int], float] = {}
        self.global_adjustment_: float = 0.0
        self.is_fitted_: bool = False

    @staticmethod
    def _require_columns(meta: pd.DataFrame):
        missing = [col for col in ["age", "health"] if col not in meta.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

    @staticmethod
    def _age_bin_mask(ages: pd.Series, age_bin: tuple[int, int]) -> pd.Series:
        start_age, end_age = age_bin
        return (ages >= start_age) & (ages < end_age)

    def fit(self, otu_train: pd.DataFrame, meta_train: pd.DataFrame):
        self._require_columns(meta_train)

        aligned_meta = meta_train.loc[otu_train.index]
        healthy_mask = aligned_meta["health"] == "y"

        if healthy_mask.sum() < 20:
            raise ValueError("Not enough healthy samples to train GAI regressor.")

        X_healthy = otu_train.loc[healthy_mask]
        y_healthy_age = aligned_meta.loc[healthy_mask, "age"].astype(float)

        self.regressor.fit(X_healthy, y_healthy_age)

        healthy_pred_age = pd.Series(
            self.regressor.predict(X_healthy),
            index=X_healthy.index,
            name="predicted_age",
        )
        healthy_raw_gai = healthy_pred_age - y_healthy_age

        self.global_adjustment_ = float(healthy_raw_gai.mean())
        self.bin_adjustments_ = {}
        healthy_ages = y_healthy_age

        for age_bin in AGE_BINS:
            mask = self._age_bin_mask(healthy_ages, age_bin)
            if mask.any():
                self.bin_adjustments_[age_bin] = float(healthy_raw_gai.loc[mask].mean())
            else:
                self.bin_adjustments_[age_bin] = self.global_adjustment_

        self.is_fitted_ = True
        return self

    def transform(self, otu: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise ValueError("GutAgingIndex must be fitted before transform.")
        self._require_columns(meta)

        aligned_meta = meta.loc[otu.index]
        ages = aligned_meta["age"].astype(float)

        predicted_age = pd.Series(
            self.regressor.predict(otu),
            index=otu.index,
            name="predicted_age",
        )
        raw_gai = predicted_age - ages

        adjustments = pd.Series(self.global_adjustment_, index=otu.index, dtype=float)
        for age_bin, adjust_value in self.bin_adjustments_.items():
            mask = self._age_bin_mask(ages, age_bin)
            adjustments.loc[mask] = adjust_value

        corrected_gai = raw_gai - adjustments
        corrected_gai.name = "gai_corrected"
        return corrected_gai
