import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_clr_transform(otu: pd.DataFrame, pseudo_count: float = 1.0) -> pd.DataFrame:
    otu_pseudo = otu + pseudo_count
    geom_mean = np.exp(np.mean(np.log(otu_pseudo), axis=1))
    return np.log(otu_pseudo.div(geom_mean, axis=0))


def compute_diversity_indices(otu: pd.DataFrame) -> pd.DataFrame:
    proportions = otu.div(otu.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    richness = (otu > 0).sum(axis=1).astype(float)

    eps = 1e-12
    safe_props = proportions.clip(lower=eps)
    shannon = -(proportions * np.log(safe_props)).sum(axis=1)
    simpson = 1.0 - (proportions**2).sum(axis=1)
    pielou = shannon / np.log(richness.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    pielou = pielou.fillna(0.0)

    return pd.DataFrame(
        {
            "div_richness": richness,
            "div_shannon": shannon,
            "div_simpson": simpson,
            "div_pielou": pielou,
        },
        index=otu.index,
    )


class MicrobiomeFeatureEngineer:
    def __init__(
        self,
        top_k_ratio_taxa: int = 10,
        network_top_taxa: int = 120,
        network_corr_threshold: float = 0.20,
        functional_top_taxa: int = 250,
        functional_components: int = 12,
    ):
        self.top_k_ratio_taxa = top_k_ratio_taxa
        self.network_top_taxa = network_top_taxa
        self.network_corr_threshold = network_corr_threshold
        self.functional_top_taxa = functional_top_taxa
        self.functional_components = functional_components
        self.ratio_pairs_: list[tuple[str, str]] = []
        self.network_taxa_: list[str] = []
        self.network_degrees_: pd.Series | None = None
        self.functional_taxa_: list[str] = []
        self.functional_pca_: PCA | None = None

    def fit(self, otu_train: pd.DataFrame):
        mean_abundance = otu_train.mean(axis=0).sort_values(ascending=False)

        ratio_taxa = mean_abundance.head(self.top_k_ratio_taxa).index.tolist()
        self.ratio_pairs_ = []
        for i in range(len(ratio_taxa)):
            for j in range(i + 1, len(ratio_taxa)):
                self.ratio_pairs_.append((ratio_taxa[i], ratio_taxa[j]))

        self.network_taxa_ = mean_abundance.head(self.network_top_taxa).index.tolist()
        clr_train = apply_clr_transform(otu_train[self.network_taxa_])
        corr = clr_train.corr().abs().fillna(0.0)
        corr_values = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_values, 0.0)

        adjacency = pd.DataFrame(
            (corr_values >= self.network_corr_threshold).astype(float),
            index=corr.index,
            columns=corr.columns,
        )
        self.network_degrees_ = adjacency.sum(axis=1)

        self.functional_taxa_ = mean_abundance.head(self.functional_top_taxa).index.tolist()
        functional_clr = apply_clr_transform(otu_train[self.functional_taxa_])
        n_components = min(self.functional_components, len(self.functional_taxa_))
        self.functional_pca_ = PCA(n_components=n_components, random_state=42)
        self.functional_pca_.fit(functional_clr)
        return self

    def _compute_ratio_features(self, otu: pd.DataFrame) -> pd.DataFrame:
        ratio_data: dict[str, pd.Series] = {}
        for left, right in self.ratio_pairs_:
            ratio_name = f"ratio_{left}_over_{right}"
            ratio_data[ratio_name] = np.log((otu[left] + 1.0) / (otu[right] + 1.0))
        return pd.DataFrame(ratio_data, index=otu.index)

    def _compute_network_features(self, otu: pd.DataFrame) -> pd.DataFrame:
        if self.network_degrees_ is None:
            raise ValueError("Feature engineer is not fitted.")

        clr_net = apply_clr_transform(otu[self.network_taxa_])
        node_weights = self.network_degrees_.reindex(self.network_taxa_).fillna(0.0)

        weighted_activity = clr_net.mul(node_weights, axis=1).sum(axis=1)
        hub_cutoff = np.percentile(node_weights.values, 80)
        hub_nodes = node_weights[node_weights >= hub_cutoff].index
        hub_activity = clr_net[hub_nodes].mean(axis=1) if len(hub_nodes) > 0 else pd.Series(0.0, index=otu.index)
        network_mean_abs = clr_net.abs().mean(axis=1)

        return pd.DataFrame(
            {
                "net_weighted_activity": weighted_activity,
                "net_hub_activity": hub_activity,
                "net_mean_abs_clr": network_mean_abs,
            },
            index=otu.index,
        )

    def _compute_functional_latent_features(self, otu: pd.DataFrame) -> pd.DataFrame:
        if self.functional_pca_ is None:
            raise ValueError("Feature engineer is not fitted.")

        functional_clr = apply_clr_transform(otu[self.functional_taxa_])
        latent = self.functional_pca_.transform(functional_clr)
        columns = [f"func_latent_{i + 1}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=otu.index, columns=columns)

    def transform(self, otu: pd.DataFrame) -> pd.DataFrame:
        clr_features = apply_clr_transform(otu)
        diversity = compute_diversity_indices(otu)
        ratio_features = self._compute_ratio_features(otu)
        network_features = self._compute_network_features(otu)
        functional_latent = self._compute_functional_latent_features(otu)

        return pd.concat(
            [clr_features, diversity, ratio_features, network_features, functional_latent],
            axis=1,
        )
