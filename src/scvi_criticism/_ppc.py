import json
import logging
import warnings
from itertools import combinations
from math import ceil
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

from ._de_utils import _get_all_de_groups, _get_top_n_genes_per_group
from ._utils import (
    _add_identity,
    _get_binary_array_from_selection,
    _get_df_corr_coeff,
    _get_df_mae,
    _get_dp_as_df,
    _get_precision_recall_f1,
)

METRIC_CV_CELL = "cv_cell"
METRIC_CV_GENE = "cv_gene"
METRIC_MWU = "mannwhitneyu"
METRIC_DIFF_EXP = "diff_exp"
DEFAULT_DE_N_TOP_GENES = 2
DEFAULT_DE_N_TOP_GENES_OVERLAP = 100

logger = logging.getLogger(__name__)


# TODO put plotting function in a separate class PPCPlot
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
        identifier = METRIC_CV_CELL if cell_wise is True else METRIC_CV_GENE
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
        """Placeholder docstring. TBD complete."""
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
        self.metrics[METRIC_MWU] = feat_df

    def plot_mwu(self, model_name: str, figure_size=None):
        """Placeholder docstring. TBD complete."""
        model_metric = self.metrics[METRIC_MWU][model_name].values
        title = f"model={model_name} | metric={METRIC_MWU} | n_cells={self.raw_counts.shape[0]}"
        figsize = figure_size if figure_size is not None else (10, 12.5)
        plt.subplots(2, 1, figsize=figsize, sharex=False)
        sns.boxplot(
            data=np.log10(model_metric),
            title=title,
        )

    def _diff_exp_compare_dotplots(self, rgg_dp_raw, rgg_dp_approx, m, kind):
        # compare the dotplots in terms of lfc/fraction values depending on `kind`

        assert kind in ["lfc", "fraction"]
        dp_kind = "color" if kind == "lfc" else "fraction"
        df_raw = _get_dp_as_df(rgg_dp_raw, dp_kind)
        df_approx = _get_dp_as_df(rgg_dp_approx, dp_kind)
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_df_raw"] = df_raw
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_df_approx"] = df_approx

        # mtr stands for metric
        mae_mtr, mae_mtr_mean = _get_df_mae(df_raw, df_approx)
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_mae"] = mae_mtr
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_mae_mean"] = mae_mtr_mean

        # some genes belong to more than one group (i.e. are markers for more than one group)
        # in this case df_raw (and same for df_approx) will have two or more columns with exactly
        # the same values. The call below removes those duplicates -- default behavior -- before computing
        # the correlation.
        pearson_mtr, pearson_mtr_mean = _get_df_corr_coeff(df_raw, df_approx, "pearson")
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_pearson"] = pearson_mtr
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_pearson_mean"] = pearson_mtr_mean

        spearman_mtr, spearman_mtr_mean = _get_df_corr_coeff(df_raw, df_approx, "spearman")
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_spearman"] = spearman_mtr
        self.metrics[METRIC_DIFF_EXP][m][f"{kind}_spearman_mean"] = spearman_mtr_mean

    def _diff_exp_compute_gene_overlaps(
        self,
        adata_raw: AnnData,
        adata_approx: AnnData,
        m: str,
        var_gene_names_col: Optional[str] = None,
        n_top_genes_overlap: Optional[int] = None,
    ):
        # compute a dataframe containing precision, recall, and f1 values that measure the overlap
        # (as described below) between the unordered set of the top `n_top_genes_overlap` of adata_raw
        # and adata_approx

        # first sanity check a few things that the code below assumes
        gene_names_raw = adata_raw.var.index if var_gene_names_col is None else adata_raw.var[var_gene_names_col]
        gene_names_approx = (
            adata_approx.var.index if var_gene_names_col is None else adata_approx.var[var_gene_names_col]
        )
        assert np.all(gene_names_raw == gene_names_approx)
        assert _get_all_de_groups(adata_raw) == _get_all_de_groups(adata_approx)

        # get the N highly scored genes from the DE result on the raw adata and approx data
        n_genes = n_top_genes_overlap or min(adata_raw.n_vars, DEFAULT_DE_N_TOP_GENES_OVERLAP)
        top_genes_raw = _get_top_n_genes_per_group(adata_raw, n_genes, var_gene_names_col)
        top_genes_approx = _get_top_n_genes_per_group(adata_approx, n_genes, var_gene_names_col)
        # get precision/recall while considering the unordered set of top ranked genes between
        # raw and approx DE results. To do that we "binarize" the gene selections: we create two
        # binary vectors (one for raw, one for approx) where a 1 in the vector means gene was
        # selected.
        groups = _get_all_de_groups(adata_raw)
        df = pd.DataFrame(index=groups, columns=["precision", "recall", "f1"], dtype=float)
        for g in groups:
            ground_truth = _get_binary_array_from_selection(gene_names_raw, top_genes_raw[g])
            pred = _get_binary_array_from_selection(gene_names_approx, top_genes_approx[g])
            assert np.sum(ground_truth) == n_genes and np.sum(pred) == n_genes
            prf = _get_precision_recall_f1(ground_truth, pred)
            df.loc[g] = prf[0], prf[1], prf[2]
        self.metrics[METRIC_DIFF_EXP][m]["gene_comparisons"] = df

    def diff_exp(
        self,
        adata_obs_raw: pd.DataFrame,
        adata_var_raw: pd.DataFrame,
        de_groupby: str,
        de_method: str = "t-test",
        var_gene_names_col: Optional[str] = None,
        n_top_genes: Optional[int] = None,
        n_top_genes_overlap: Optional[int] = None,
    ):
        """Placeholder docstring. TBD complete."""
        # run DE with the raw counts
        adata_raw = AnnData(X=self.raw_counts.tocsr(), obs=adata_obs_raw, var=adata_var_raw)
        norm_sum = 1e4
        sc.pp.normalize_total(adata_raw, target_sum=norm_sum)
        sc.pp.log1p(adata_raw)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            sc.tl.rank_genes_groups(adata_raw, de_groupby, use_raw=False, method=de_method)

        # get the N highly scored genes from the DE result on the raw data
        n_genes = n_top_genes if n_top_genes is not None else DEFAULT_DE_N_TOP_GENES
        var_names = _get_top_n_genes_per_group(adata_raw, n_genes, var_gene_names_col)

        self.metrics[METRIC_DIFF_EXP] = {}
        self.metrics[METRIC_DIFF_EXP]["adata_raw"] = adata_raw.copy()
        self.metrics[METRIC_DIFF_EXP]["var_names"] = var_names

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

        # get posterior predictive samples from the model (aka approx counts)
        pp_samples = self.posterior_predictive_samples.items()
        for m, samples in pp_samples:
            adata_approx = AnnData(X=csr_matrix(samples), obs=adata_obs_raw, var=adata_var_raw)
            sc.pp.normalize_total(adata_approx, target_sum=norm_sum)
            sc.pp.log1p(adata_approx)

            # run DE with the approx counts
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
                sc.tl.rank_genes_groups(adata_approx, de_groupby, use_raw=False, method=de_method)

            self.metrics[METRIC_DIFF_EXP][m] = {}
            self.metrics[METRIC_DIFF_EXP][m]["adata_approx"] = adata_approx.copy()

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

            self._diff_exp_compare_dotplots(rgg_dp_raw, rgg_dp_approx, m, "lfc")
            self._diff_exp_compare_dotplots(rgg_dp_raw, rgg_dp_approx, m, "fraction")

            self._diff_exp_compute_gene_overlaps(adata_raw, adata_approx, m, var_gene_names_col, n_top_genes_overlap)

    def _plot_diff_exp_scatter_plots(
        self,
        title: str,
        df_1,
        df_2,
        mae: pd.Series,
        pearson: pd.Series,
        spearman: pd.Series,
        figure_size=None,
    ):
        # define subplot grid
        # https://engineeringfordatascience.com/posts/matplotlib_subplots/
        ncols = 4
        nrows = ceil(len(df_1.index) / ncols)
        figsize = figure_size if figure_size is not None else (20, 3 * nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained")
        fig.suptitle(title, fontsize=18)
        axs_lst = axs.ravel()
        # https://stackoverflow.com/a/66799199
        for ax in axs_lst:
            ax.set_axis_off()
        # plot all
        i = 0
        for group in df_1.index:  # TODO allow to plot a subset of the groups?
            ax = axs_lst[i]
            i += 1
            raw = df_1.loc[group]
            approx = df_2.loc[group]
            ax.scatter(raw, approx)
            _add_identity(ax, color="r", ls="--", alpha=0.5)
            ax.set_title(
                f"{group} \n pearson={pearson[group]:.2f} - spearman={spearman[group]:.2f} - mae={mae[group]:.2f}"
            )
            ax.set_axis_on()
        plt.show()

    def plot_diff_exp(
        self,
        model_name: str,
        var_gene_names_col: Optional[str] = None,
        var_names_subset: Optional[Sequence[str]] = None,
        plot_kind: str = "dotplots",
        figure_size=None,
    ):
        """Placeholder docstring. TBD complete."""
        assert plot_kind in ["dotplots", "lfc_comparisons", "fraction_comparisons", "gene_overlaps", "summary"]

        adata_approx = self.metrics[METRIC_DIFF_EXP][model_name]["adata_approx"]
        adata_raw = self.metrics[METRIC_DIFF_EXP]["adata_raw"]
        var_names = self.metrics[METRIC_DIFF_EXP]["var_names"]
        if var_names_subset is not None:
            var_names = {k: v for k, v in var_names.items() if k in var_names_subset}

        if plot_kind == "dotplots":
            # plot dotplots for raw and approx
            sc.pl.rank_genes_groups_dotplot(
                adata_raw,
                values_to_plot="logfoldchanges",
                vmax=7,
                vmin=-7,
                cmap="bwr",
                dendrogram=False,
                gene_symbols=var_gene_names_col,
                var_names=var_names,
            )

            # plot, using var_names, i.e., the N highly scored genes from the DE result on adata_raw
            # we do this because the N highly scored genes (per group) in the adata_approx are not necessarily
            # the same as adata_raw. this discrepancy is evaluated elsewhere (when looking at gene overlaps)
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
        elif plot_kind == "lfc_comparisons" or plot_kind == "fraction_comparisons":
            kind = "lfc" if plot_kind == "lfc_comparisons" else "fraction"
            df_raw = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_df_raw"]
            df_approx = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_df_approx"]
            mae_mtr = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_mae"]
            pearson_mtr = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_pearson"]
            spearman_mtr = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_spearman"]
            mae_mtr_mean = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_mae_mean"]
            pearson_mtr_mean = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_pearson_mean"]
            spearman_mtr_mean = self.metrics[METRIC_DIFF_EXP][model_name][f"{kind}_spearman_mean"]
            # log mae, pearson corr, spearman corr
            if plot_kind == "lfc_comparisons":
                desc = "LFC (1 vs all) gene expressions across groups"
            else:
                desc = "fractions of genes expressed per group across groups"
            logger.info(
                f"{desc}:\n"
                f"Mean Absolute Error={mae_mtr_mean:.2f},\n"
                f"Pearson correlation={pearson_mtr_mean:.2f}\n"
                f"Spearman correlation={spearman_mtr_mean:.2f}"
            )
            title = f"{desc}, x=raw DE, y=approx DE, red line=identity"
            self._plot_diff_exp_scatter_plots(
                title, df_raw, df_approx, mae_mtr, pearson_mtr, spearman_mtr, figure_size=figure_size
            )
        elif plot_kind == "gene_overlaps":
            # plot per-group F1 bar plots for the given n_genes
            gene_comparisons = self.metrics[METRIC_DIFF_EXP][model_name]["gene_comparisons"]
            mean_f1 = np.mean(gene_comparisons["f1"])
            figsize = figure_size if figure_size is not None else (0.8 * len(gene_comparisons), 3)
            gene_comparisons.plot.bar(
                y="f1",
                figsize=figsize,
                title=f"Gene overlap F1 scores across groups - mean_f1 = {mean_f1:.2f}",
                legend=False,
                layout="constrained",
            )
        elif plot_kind == "summary":
            lfc_pearson = self.metrics[METRIC_DIFF_EXP][model_name]["lfc_pearson"]
            lfc_spearman = self.metrics[METRIC_DIFF_EXP][model_name]["lfc_spearman"]
            fraction_pearson = self.metrics[METRIC_DIFF_EXP][model_name]["fraction_pearson"]
            fraction_spearman = self.metrics[METRIC_DIFF_EXP][model_name]["fraction_spearman"]
            gene_comparisons = self.metrics[METRIC_DIFF_EXP][model_name]["gene_comparisons"]["f1"]

            # sanity check all indices are the same
            idxs = []
            idxs.append(lfc_pearson.index)
            idxs.append(lfc_spearman.index)
            idxs.append(fraction_pearson.index)
            idxs.append(fraction_spearman.index)
            idxs.append(gene_comparisons.index)
            for couple in combinations(idxs, 2):
                assert couple[0].equals(couple[1])

            cols = ["lfc_pearson", "lfc_spearman", "gene_frac_pearson", "gene_frac_spearman", "gene_overlap_f1"]
            summary_df = pd.DataFrame(index=lfc_pearson.index, columns=cols)
            summary_df["lfc_pearson"] = lfc_pearson
            summary_df["lfc_spearman"] = lfc_spearman
            summary_df["gene_frac_pearson"] = fraction_pearson
            summary_df["gene_frac_spearman"] = fraction_spearman
            summary_df["gene_overlap_f1"] = gene_comparisons

            summary_df.boxplot(figsize=(10, 8))
        else:
            raise ValueError("Unknown plot_kind: {plot_kind}")


def run_ppc(
    adata: AnnData,
    model,
    metric: str,
    n_samples: int,
    layer: Optional[str] = None,
    custom_indices: Optional[Union[int, Sequence[int]]] = None,
    **metric_specific_kwargs,
):
    """Compute the given PPC metric for the given model, data and indices."""
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

    # calculate metrics
    if (metric == METRIC_CV_CELL) or (metric == METRIC_CV_GENE):
        sp.coefficient_of_variation(cell_wise=(metric == METRIC_CV_CELL))
    elif metric == METRIC_MWU:
        sp.mann_whitney_u()
    elif metric == METRIC_DIFF_EXP:
        # adata.obs is needed for de_groupby
        sp.diff_exp(adata[indices].obs, adata.var, **metric_specific_kwargs)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

    return sp
