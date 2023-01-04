import logging
from itertools import combinations
from math import ceil
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from ._constants import METRIC_CV_CELL, METRIC_CV_GENE, METRIC_DIFF_EXP, METRIC_MWU
from ._ppc import PPC
from ._utils import _add_identity

logger = logging.getLogger(__name__)


class PPCPlot:
    """Plotting utilities for posterior predictive checks"""

    def __init__(
        self,
        ppc: PPC,
    ):
        self._ppc = ppc

    def plot_cv(self, model_name: str, cell_wise: bool = True):
        """Placeholder docstring. TBD complete."""
        metric = METRIC_CV_CELL if cell_wise is True else METRIC_CV_GENE
        model_metric = self._ppc.metrics[metric][model_name].values
        raw_metric = self._ppc.metrics[metric]["Raw"].values
        title = f"model={model_name} | metric={metric} | n_cells={self._ppc.raw_counts.shape[0]}"

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

    def plot_mwu(self, model_name: str, figure_size=None):
        """Placeholder docstring. TBD complete."""
        model_metric = self._ppc.metrics[METRIC_MWU][model_name].values
        title = f"model={model_name} | metric={METRIC_MWU} | n_cells={self._ppc.raw_counts.shape[0]}"
        figsize = figure_size if figure_size is not None else (10, 12.5)
        plt.subplots(2, 1, figsize=figsize, sharex=False)
        sns.boxplot(
            data=np.log10(model_metric),
            title=title,
        )

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
        de_metrics = self._ppc.metrics[METRIC_DIFF_EXP]
        model_de_metrics = de_metrics[model_name]

        if plot_kind == "dotplots":
            # plot dotplots for raw and approx
            adata_raw = de_metrics["adata_raw"]
            var_names = de_metrics["var_names"]
            if var_names_subset is not None:
                var_names = {k: v for k, v in var_names.items() if k in var_names_subset}
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
            adata_approx = model_de_metrics["adata_approx"]
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
            df_raw = de_metrics[f"{kind}_df_raw"]
            df_approx = model_de_metrics[f"{kind}_df_approx"]
            mae_mtr = model_de_metrics[f"{kind}_mae"]
            pearson_mtr = model_de_metrics[f"{kind}_pearson"]
            spearman_mtr = model_de_metrics[f"{kind}_spearman"]
            mae_mtr_mean = model_de_metrics[f"{kind}_mae_mean"]
            pearson_mtr_mean = model_de_metrics[f"{kind}_pearson_mean"]
            spearman_mtr_mean = model_de_metrics[f"{kind}_spearman_mean"]
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
            gene_comparisons = model_de_metrics["gene_comparisons"]
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
            lfc_pearson = model_de_metrics["lfc_pearson"]
            lfc_spearman = model_de_metrics["lfc_spearman"]
            fraction_pearson = model_de_metrics["fraction_pearson"]
            fraction_spearman = model_de_metrics["fraction_spearman"]
            gene_comparisons = model_de_metrics["gene_comparisons"]["f1"]

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
            plt.show()
        else:
            raise ValueError("Unknown plot_kind: {plot_kind}")
