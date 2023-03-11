import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

from ._constants import METRIC_CV_CELL, METRIC_CV_GENE, METRIC_DIFF_EXP
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
        Plot coefficient of variation metrics results.

        See our tutorials for a demonstration of the generated plot along with detailed explanations.

        Parameters
        ----------
        model_name
            Name of the model
        cell_wise
            Whether to plot the cell-wise or gene-wise metric
        """
        metric = METRIC_CV_CELL if cell_wise is True else METRIC_CV_GENE
        model_metric = self._ppc.metrics[metric][model_name].values
        raw_metric = self._ppc.metrics[metric]["Raw"].values
        title = f"model={model_name} | metric={metric} | n_cells={self._ppc.raw_counts.shape[0]}"

        # log mae, pearson corr, spearman corr, R^2
        logger.info(
            f"{title}:\n"
            f"Mean Absolute Error={mae(model_metric, raw_metric):.2f},\n"
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

    def plot_diff_exp(
        self,
        model_name: str,
        plot_kind: str,
        figure_size=None,
    ):
        """
        Plot differential expression results.

        Parameters
        ----------
        model_name
            Name of the model
        var_gene_names_col
            Column name in the `adata.var` attribute containing the gene names, if different from `adata.var_names`
        figure_size
            Size of the figure to plot. If None, we will use a heuristic to determine the figure size.
        """
        de_metrics = self._ppc.metrics[METRIC_DIFF_EXP]
        model_de_metrics = de_metrics[de_metrics["model"] == model_name]
        del model_de_metrics["lfc_mae"]  # not on the same scale as the other ones

        bar_colors = {
            "lfc_pearson": "#AA8FC4",
            "lfc_spearman": "#A4DE87",
            "pr_auc": "#D9A5CC",
            "roc_auc": "#769FCC",
        }
        if plot_kind == "summary_violin":
            col_names = {
                "lfc_pearson": "LFC\nPearson",
                "lfc_spearman": "LFC\nSpearman",
                "pr_auc": "PR\nAUC",
                "roc_auc": "ROC\nAUC",
            }
            model_de_metrics.rename(columns=col_names, inplace=True)
            sns.set(rc={"figure.figsize": (6, 6)})
            sns.set_theme(style="white")
            sns.violinplot(model_de_metrics, palette=sns.color_palette("pastel"))
            plt.grid()
        elif plot_kind == "summary_barv":
            figsize = figure_size if figure_size is not None else (0.8 * len(model_de_metrics), 3)
            model_de_metrics.index.name = None
            model_de_metrics.plot.bar(figsize=figsize, color=bar_colors, edgecolor="black")
            plt.legend(loc="lower right")
            # add a grid and hide it behind plot objects.
            # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
            ax = plt.gca()
            ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
            ax.set(axisbelow=True)
        elif plot_kind == "summary_barh":
            figsize = figure_size if figure_size is not None else (3, 0.8 * len(model_de_metrics))
            model_de_metrics.index.name = None
            model_de_metrics.plot.barh(figsize=figsize, color=bar_colors, edgecolor="black")
            plt.legend(loc="upper right")
            # add a grid and hide it behind plot objects.
            # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
            ax = plt.gca()
            ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
            ax.set(axisbelow=True)
        else:
            raise ValueError("Unknown plot_kind: {plot_kind}")
