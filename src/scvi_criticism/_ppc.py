import json
import logging
import warnings
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from ._utils import _add_identity, _get_df_corr_coeff, _get_df_mae, _get_dp_as_df

METRIC_CV_CELL = "cv_cell"
METRIC_CV_GENE = "cv_gene"
METRIC_MWU = "mannwhitneyu"
METRIC_DIFF_EXP = "diff_exp"

logger = logging.getLogger(__name__)


class PPC:
    """Posterior predictive checks for comparing single-cell generative models"""

    def __init__(
        self,
        n_samples: int = 1,
        raw_counts: Optional[Union[np.ndarray, csr_matrix, coo_matrix]] = None,
    ):
        if isinstance(raw_counts, np.ndarray):
            self.raw_counts = coo_matrix(raw_counts)
        elif isinstance(raw_counts, csr_matrix):
            self.raw_counts = raw_counts.tocoo()
        elif isinstance(raw_counts, coo_matrix):
            self.raw_counts = raw_counts
        else:
            self.raw_counts = None
        self.posterior_predictive_samples = {}
        self.n_samples = n_samples
        self.models = {}
        self.metrics = {}

    def __repr__(self) -> str:
        return (
            f"--- Posterior Predictive Checks ---\n"
            f"n_samples = {self.n_samples}\n"
            f"raw_counts shape = {self.raw_counts.shape}\n"
            f"models: {list(self.models.keys())}\n"
            f"metrics: \n{self._metrics_repr()}"
        )

    def _metrics_repr(self) -> str:
        def custom_handle_unserializable(o):
            if isinstance(o, AnnData):
                return f"AnnData with n_obs={o.n_obs}, n_vars={o.n_vars}"
            return f"ERROR unserializable type: {type(o)}"

        return json.dumps(self.metrics, indent=4, default=custom_handle_unserializable)

    def store_posterior_predictive_samples(
        self,
        models_dict,
        batch_size=32,
        indices=None,
    ):
        """Gathers posterior predictive samples."""
        self.models = models_dict
        self.batch_size = batch_size
        # first_model = next(iter(models_dict.keys()))
        # self.dataset = models_dict[first_model].adata

        for m, model in self.models.items():
            pp_counts = model.posterior_predictive_sample(
                model.adata,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                indices=indices,
            )
            self.posterior_predictive_samples[m] = pp_counts

    def coefficient_of_variation(self, cell_wise: bool = True):
        """
        Calculate the coefficient of variation.

        Parameters:
            cell_wise: Calculate for each cell across genes if True, else do the reverse.
        """
        axis = 1 if cell_wise is True else 0
        identifier = "cv_cell" if cell_wise is True else "cv_gene"
        df = pd.DataFrame()
        pp_samples = self.posterior_predictive_samples.items()
        for m, samples in pp_samples:
            cv = np.nanmean(
                np.std(samples, axis=axis) / np.mean(samples, axis=axis),
                axis=-1,
            )

            df[m] = cv.ravel()
            df[m] = np.nan_to_num(df[m])

        raw = self.raw_counts.todense()
        df["Raw"] = pd.DataFrame(
            np.asarray(np.std(raw, axis=axis)).squeeze() / np.asarray(np.mean(raw, axis=axis)).squeeze()
        )
        df["Raw"] = np.nan_to_num(df["Raw"])

        self.metrics[identifier] = df

    def plot_cv(self, model_name: str, cell_wise: bool = True):
        """Placeholder docstring. TODO complete"""
        metric = METRIC_CV_CELL if cell_wise is True else METRIC_CV_GENE
        model_metric = self.metrics[metric][model_name].values
        raw_metric = self.metrics[metric]["Raw"].values
        title = f"model={model_name} | metric={metric} | n_cells={self.raw_counts.shape[0]}"

        # log mae, mse, pearson corr, spearman corr
        logger.info(
            f"{title}:\n"
            f"Mean Absolute Error={mae(model_metric, raw_metric):.2f},\n"
            f"Mean Squared Error={mse(model_metric, raw_metric):.2f}\n"
            f"Pearson correlation={pearsonr(model_metric, raw_metric)[0]:.2f}\n"
            f"Spearman correlation={spearmanr(model_metric, raw_metric)[0]:.2f}\n"
        )

        # plot visual correlation (scatter plot)
        plt.scatter(model_metric, raw_metric)
        ax = plt.gca()
        _add_identity(ax, color="r", ls="--", alpha=0.5)
        plt.xlabel("model")
        plt.ylabel("raw")
        plt.title(title)
        plt.show()

    def mann_whitney_u(self):
        """Calculate the Mann–Whitney U statistic."""
        feat_df = pd.DataFrame()
        pp_samples = self.posterior_predictive_samples.items()
        raw = self.raw_counts.todense()
        for m, samples in pp_samples:
            sam = samples
            feats = []
            for g in range(samples.shape[1]):
                Us = []
                for n in range(samples.shape[2]):
                    U, _ = mannwhitneyu(sam[:, g, n], raw[:, g])
                    Us.append(U)
                feats.append(np.mean(Us))
            to_add = feats
            if len(to_add) != raw.shape[1]:
                raise ValueError()
            feat_df[m] = to_add
        self.metrics["mannwhitneyu"] = feat_df

    def plot_mwu(self, model_name: str):
        """Placeholder docstring. TODO complete"""
        model_metric = self.metrics[METRIC_MWU][model_name].values
        title = f"model={model_name} | metric={METRIC_MWU} | n_cells={self.raw_counts.shape[0]}"
        plt.subplots(2, 1, figsize=(10, 12.5), sharex=False)
        sns.boxplot(
            data=np.log10(model_metric),
            title=title,
        )

    def diff_exp(
        self,
        adata_obs_raw: pd.DataFrame,
        adata_var_raw: pd.DataFrame,
        de_groupby: str,
        de_method: str = "t-test",
        var_gene_names_col: Optional[str] = None,
    ):
        """Placeholder docstring. TODO complete"""
        # run DE with the raw counts
        adata_raw = AnnData(X=self.raw_counts.tocsr(), obs=adata_obs_raw, var=adata_var_raw)
        norm_sum = 1e4
        sc.pp.normalize_total(adata_raw, target_sum=norm_sum)
        sc.pp.log1p(adata_raw)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            sc.tl.rank_genes_groups(adata_raw, de_groupby, use_raw=False, method=de_method)

        # get the N highly scored genes from the DE result on the raw data
        rgg = adata_raw.uns["rank_genes_groups"]
        rgg_names = pd.DataFrame.from_records(rgg["names"])
        n_genes = 2  # TODO parametrize?
        var_names = {}
        for group in rgg_names.columns:
            if var_gene_names_col is not None:
                var_names[group] = adata_raw.var.loc[rgg_names[group].values[:n_genes]]["gene_names"].values.tolist()
            else:
                var_names[group] = rgg_names[group].values[:n_genes].tolist()

        self.metrics["diff_exp"] = {}
        self.metrics["diff_exp"]["adata_raw"] = adata_raw.copy()
        self.metrics["diff_exp"]["var_names"] = var_names

        # get the dotplot values for adata_raw
        rgg_dp_raw = sc.pl.rank_genes_groups_dotplot(
            adata_raw,
            values_to_plot="logfoldchanges",
            vmax=7,
            vmin=-7,
            cmap="bwr",
            dendrogram=False,
            gene_symbols=var_gene_names_col,
            var_names=var_names,
            return_fig=True,
        )

        # get posterior predictive samples from the model (aka approx. counts)
        pp_samples = self.posterior_predictive_samples.items()
        for m, samples in pp_samples:
            adata_approx = AnnData(X=csr_matrix(samples), obs=adata_obs_raw, var=adata_var_raw)
            sc.pp.normalize_total(adata_approx, target_sum=norm_sum)
            sc.pp.log1p(adata_approx)

            # run DE with the approx. counts
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
                sc.tl.rank_genes_groups(adata_approx, de_groupby, use_raw=False, method=de_method)

            self.metrics["diff_exp"][m] = {}
            self.metrics["diff_exp"][m]["adata_approx"] = adata_approx.copy()

            rgg_dp_approx = sc.pl.rank_genes_groups_dotplot(
                adata_approx,
                values_to_plot="logfoldchanges",
                vmax=7,
                vmin=-7,
                cmap="bwr",
                dendrogram=False,
                gene_symbols=var_gene_names_col,
                var_names=var_names,
                return_fig=True,
            )

            # compare the dotplots in terms of lfc and fraction values
            lfc_df_raw = _get_dp_as_df(rgg_dp_raw)
            lfc_df_approx = _get_dp_as_df(rgg_dp_approx)
            lfc_mae, lfc_mae_mean = _get_df_mae(lfc_df_approx, lfc_df_raw)
            lfc_pearson, lfc_pearson_mean = _get_df_corr_coeff(lfc_df_approx, lfc_df_raw, "pearson")
            lfc_spearman, lfc_spearman_mean = _get_df_corr_coeff(lfc_df_approx, lfc_df_raw, "spearman")
            self.metrics[METRIC_DIFF_EXP][m]["lfc_pearson"] = lfc_pearson
            self.metrics[METRIC_DIFF_EXP][m]["lfc_spearman"] = lfc_spearman
            self.metrics[METRIC_DIFF_EXP][m]["lfc_mae"] = lfc_mae
            self.metrics[METRIC_DIFF_EXP][m]["lfc_pearson_mean"] = lfc_pearson_mean
            self.metrics[METRIC_DIFF_EXP][m]["lfc_spearman_mean"] = lfc_spearman_mean
            self.metrics[METRIC_DIFF_EXP][m]["lfc_mae_mean"] = lfc_mae_mean

    def plot_diff_exp(
        self,
        model_name: str,
        var_gene_names_col: Optional[str] = None,
        var_names_subset: Optional[Sequence[str]] = None,
    ):
        """Placeholder docstring. TODO complete"""
        adata_approx = self.metrics[METRIC_DIFF_EXP][model_name]["adata_approx"]
        adata_raw = self.metrics[METRIC_DIFF_EXP]["adata_raw"]
        var_names = self.metrics[METRIC_DIFF_EXP]["var_names"]
        if var_names_subset is not None:
            var_names = {k: v for k, v in var_names.items() if k in var_names_subset}

        # TODO add plot title
        sc.pl.rank_genes_groups_dotplot(
            adata_raw,
            values_to_plot="logfoldchanges",
            # min_logfoldchange=3,
            vmax=7,
            vmin=-7,
            cmap="bwr",
            dendrogram=False,
            gene_symbols=var_gene_names_col,
            var_names=var_names,
        )

        # plot, using var_names, i.e., the N highly scored genes from the DE result on adata_raw
        # we do this because the N highly scored genes (per group) in the adata_approx are not the same as adata_raw. this
        # discrepancy is evaluated elsewhere
        sc.pl.rank_genes_groups_dotplot(
            adata_approx,
            values_to_plot="logfoldchanges",
            vmax=7,
            vmin=-7,
            cmap="bwr",
            dendrogram=False,
            gene_symbols=var_gene_names_col,
            var_names=var_names,
        )


def run_ppc(
    adata: AnnData,
    model,
    metric: str,
    n_samples: int,
    layer: Optional[str] = None,
    custom_indices: Optional[Union[int, Sequence[int]]] = None,
    do_plot: bool = True,
    **metric_specific_kwargs,
):
    """Compute the given PPC metric for the given model, data and indices. Plot results by default"""
    # determine indices to use
    if isinstance(custom_indices, list):
        indices = custom_indices
    elif isinstance(custom_indices, int):
        indices = np.random.randint(0, adata.n_obs, custom_indices)
    else:
        indices = np.arange(adata.n_obs)

    # create PPC instance and compute pp samples
    raw_data = adata[indices].X if layer is None else adata[indices].layers[layer]
    sp = PPC(n_samples=n_samples, raw_counts=raw_data)
    model_name = f"{model.__class__.__name__}"
    models_dict = {model_name: model}
    sp.store_posterior_predictive_samples(models_dict, indices=indices)

    # calculate metrics and plot if asked to
    if (metric == METRIC_CV_CELL) or (metric == METRIC_CV_GENE):
        cw = metric == "cv_cell"
        sp.coefficient_of_variation(cell_wise=cw)
        if do_plot:
            sp.plot_cv(model_name, cell_wise=cw)
    elif metric == METRIC_MWU:
        sp.mann_whitney_u()
        if do_plot:
            sp.plot_mwu(model_name)
    elif metric == METRIC_DIFF_EXP:
        # adata.obs is needed for de_groupby
        sp.diff_exp(adata[indices].obs, adata.var, **metric_specific_kwargs)
        if do_plot:
            sp.plot_diff_exp(model_name, **metric_specific_kwargs)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

    return sp
