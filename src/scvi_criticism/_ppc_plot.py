import logging
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from ._constants import (METRIC_CV_CELL, METRIC_CV_GENE, METRIC_DIFF_EXP,
                         METRIC_MWU)
from ._ppc import PPC
from ._utils import _add_identity

logger = logging.getLogger(__name__)


class PPCPlot:
    """
    Plotting utilities for posterior predictive checks

    Parameters
    ----------
    ppc
        An instance of the :class:`~scvi_criticism.PPC` class containing the computed metrics
    """

    def __init__(
        self,
        ppc: PPC,
    ):
        self._ppc = ppc

    def plot_cv(self, model_name: str, cell_wise: bool = True, plt_type: Literal["scatter", "hist2d"] = "hist2d"):
        """
        Plot the coefficient of variation metrics results.

        See our tutorials for a demonstration of the generated plot along with detailed explanations.

        Parameters
        ----------
        model_name
            Name of the model
        cell_wise
            Whether to plot the cell-wise or gene-wise metric
        plt_type
            The type of plot to generate.
        """
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
            f"r^2={r2_score(raw_metric, model_metric):.2f}\n"
        )

        # plot visual correlation (scatter plot or 2D histogram)
        if plt_type == "scatter":
            plt.scatter(model_metric, raw_metric)
        elif plt_type == "hist2d":
            h, _, _, _ = plt.hist2d(model_metric, raw_metric, bins=300)
            plt.close()  # don't show it yet
            a = h.flatten()
            cmin = np.min(a[a > 0])  # the smallest value > 0
            h = plt.hist2d(model_metric, raw_metric, bins=300, cmin=cmin, rasterized=True)
        else:
            raise ValueError(f"Invalid plt_type={plt_type}")
        ax = plt.gca()
        _add_identity(ax, color="r", ls="--", alpha=0.5)
        # add line of best fit
        # a, b = np.polyfit(model_metric, raw_metric, 1)
        # plt.plot(model_metric, a*model_metric+b)
        # add labels and titles
        plt.xlabel("model")
        plt.ylabel("raw")
        plt.title(title)

    def plot_mwu(self, model_name: str, figure_size=None):
        """
        Plot the Mann-Whitney U test results.

        See our tutorials for a demonstration of the generated plot along with detailed explanations.

        Parameters
        ----------
        model_name
            Name of the model
        figure_size
            Size of the figure to plot
        """
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
        save_fig: bool = False,
    ):
        """
        Plot the differential expression results.

        See our tutorials for a demonstration of the generated plot along with detailed explanations.

        Parameters
        ----------
        model_name
            Name of the model
        var_gene_names_col
            Column name in the `adata.var` attribute containing the gene names, if different from `adata.var_names`
        var_names_subset
            Subset of the variable names to plot
        plot_kind
            Kind of plot to plot (e.g., dotplots, gene_overlaps, summary, etc.). See the tutorial for a detailed
            explanation of the different kinds of plots.
        figure_size
            Size of the figure to plot. If None, we will use a heuristic to determine the figure size.
        save_fig
            Whether to save the figure to a file. The path(s) to the saved figures will be returned.
        """
        assert plot_kind in [
            "dotplots",
            "lfc_comparisons",
            "fraction_comparisons",
            "gene_overlaps",
            "summary_box",
            "summary_barh",
            "summary_barv",
            "summary_return_df",
        ]
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
                save="raw.svg" if save_fig else None,
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
                save="approx.svg" if save_fig else None,
            )

            # return absolute fig Paths if applicable
            if save_fig:
                # I wish scanpy had a better way than to do this...
                scanpy_fig_path = "figures/dotplot_{}"
                return (
                    Path(scanpy_fig_path.format("raw.svg")).resolve(),
                    Path(scanpy_fig_path.format("approx.svg")).resolve(),
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
                desc = "fractions of cells expressing each gene in each group"
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
        elif plot_kind.startswith("summary_"):
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

            cols = ["lfc_pearson", "lfc_spearman", "cell_frac_pearson", "cell_frac_spearman", "gene_overlap_f1"]
            summary_df = pd.DataFrame(index=lfc_pearson.index, columns=cols)
            summary_df["lfc_pearson"] = lfc_pearson
            summary_df["lfc_spearman"] = lfc_spearman
            summary_df["cell_frac_pearson"] = fraction_pearson
            summary_df["cell_frac_spearman"] = fraction_spearman
            summary_df["gene_overlap_f1"] = gene_comparisons

            bar_colors = {
                "lfc_pearson": "#AA8FC4",
                "lfc_spearman": "#A4DE87",
                "cell_frac_pearson": "#F7E6AD",
                "cell_frac_spearman": "#D9A5CC",
                "gene_overlap_f1": "#769FCC",
            }
            if plot_kind == "summary_return_df":
                return summary_df
            elif plot_kind == "summary_box":
                # summary_df.boxplot(figsize=(10, 8))
                col_names = {
                    "lfc_pearson": "LFC\nPearson",
                    "lfc_spearman": "LFC\nSpearman",
                    "cell_frac_pearson": "Cell Frac\nPearson",
                    "cell_frac_spearman": "Cell Frac\nSpearman",
                    "gene_overlap_f1": "Gene Overlap\nF1",
                }
                summary_df.rename(columns=col_names, inplace=True)
                sns.set(rc={"figure.figsize": (6, 6)})
                sns.set_theme(style="white")
                sns.violinplot(summary_df, palette=sns.color_palette("pastel"))
                plt.grid()
            elif plot_kind == "summary_barv":
                figsize = figure_size if figure_size is not None else (0.8 * len(summary_df), 3)
                summary_df.index.name = None
                summary_df.plot.bar(figsize=figsize, color=bar_colors, edgecolor="black")
                plt.legend(loc="lower right")
                # add a grid and hide it behind plot objects.
                # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
                ax = plt.gca()
                ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
                ax.set(axisbelow=True)
            else:  # summary_barh
                figsize = figure_size if figure_size is not None else (3, 0.8 * len(summary_df))
                summary_df.index.name = None
                summary_df.plot.barh(figsize=figsize, color=bar_colors, edgecolor="black")
                plt.legend(loc="upper right")
                # add a grid and hide it behind plot objects.
                # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
                ax = plt.gca()
                ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
                ax.set(axisbelow=True)
        else:
            raise ValueError("Unknown plot_kind: {plot_kind}")
