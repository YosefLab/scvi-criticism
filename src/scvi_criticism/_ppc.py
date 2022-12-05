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

        # get posterior predictive samples from the model (aka approx. counts)
        self.metrics["diff_exp"]["adata_approx"] = {}
        pp_samples = self.posterior_predictive_samples.items()
        for m, samples in pp_samples:
            adata_approx = AnnData(X=csr_matrix(samples), obs=adata_obs_raw, var=adata_var_raw)
            sc.pp.normalize_total(adata_approx, target_sum=norm_sum)
            sc.pp.log1p(adata_approx)

            # run DE with the approx. counts
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
                sc.tl.rank_genes_groups(adata_approx, de_groupby, use_raw=False, method=de_method)

            self.metrics["diff_exp"]["adata_approx"][m] = adata_approx.copy()


def _get_ppc_metrics(
    adata: AnnData,
    model,
    metric: str,
    n_samples: int,
    indices: Sequence[int],
    layer: Optional[str] = None,
    **metric_specific_kwargs,
) -> None:
    raw_data = adata[indices, :].X if layer is None else adata[indices, :].layers[layer]

    sp = PPC(n_samples=n_samples, raw_counts=raw_data)
    model_name = f"{model.__class__.__name__}"
    models_dict = {model_name: model}
    sp.store_posterior_predictive_samples(models_dict, indices=indices)

    model_metric, raw_metric = None, None
    if (metric == METRIC_CV_CELL) or (metric == METRIC_CV_GENE):
        sp.coefficient_of_variation(cell_wise=(metric == "cv_cell"))
        model_metric = sp.metrics[metric][model_name].values
        raw_metric = sp.metrics[metric]["Raw"].values
    elif metric == METRIC_MWU:
        sp.mann_whitney_u()
        model_metric = sp.metrics[metric][model_name].values
        raw_metric = None
    elif metric == METRIC_DIFF_EXP:
        # adata.obs is needed for de_groupby
        sp.diff_exp(adata.obs, adata.var, **metric_specific_kwargs)
        model_metric = sp.metrics[metric]["adata_approx"][model_name]
        raw_metric = sp.metrics[metric]["adata_raw"]
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

    return sp, model_metric, raw_metric


def _plot_ppc_cv(title: str, model_metric, raw_metric):
    # from https://stackoverflow.com/a/28216751
    def add_identity(axes, *line_args, **line_kwargs):
        (identity,) = axes.plot([], [], *line_args, **line_kwargs)

        def callback(axes):
            low_x, high_x = axes.get_xlim()
            low_y, high_y = axes.get_ylim()
            low = max(low_x, low_y)
            high = min(high_x, high_y)
            identity.set_data([low, high], [low, high])

        callback(axes)
        axes.callbacks.connect("xlim_changed", callback)
        axes.callbacks.connect("ylim_changed", callback)
        return axes

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
    add_identity(ax, color="r", ls="--", alpha=0.5)
    plt.xlabel("model")
    plt.ylabel("raw")
    plt.title(title)
    plt.show()


def _plot_ppc_mwu(title: str, model_metric):
    _, ax = plt.subplots(2, 1, figsize=(10, 12.5), sharex=False)
    sns.boxplot(
        data=np.log10(model_metric),
        title=title,
    )


def _plot_ppc_diff_exp(title: str, model_metric: AnnData, raw_metric: AnnData):
    # sc.pl.rank_genes_groups_dotplot(
    #     raw_metric,
    #     values_to_plot='logfoldchanges',
    #     # min_logfoldchange=3,
    #     vmax=7,
    #     vmin=-7,
    #     cmap='bwr',
    #     dendrogram=False,
    #     gene_symbols="gene_names",
    #     var_names=var_names,
    # )
    pass


def plot_ppc(title: str, model_metric, raw_metric, metric: str, sp):
    """Plot and log summary ppc results for the given model and raw metric vectors"""
    if (metric == METRIC_CV_CELL) or (metric == METRIC_CV_GENE):
        _plot_ppc_cv(title, model_metric, raw_metric)
    elif metric == METRIC_MWU:
        _plot_ppc_mwu(title, model_metric)
    elif metric == METRIC_DIFF_EXP:
        _plot_ppc_diff_exp(title, model_metric, raw_metric, sp)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")


def run_ppc(
    adata: AnnData,
    model,
    metric: str,
    n_samples: int,
    custom_indices: Optional[Sequence[int]] = None,
    n_indices: Optional[int] = None,
    layer: Optional[str] = None,
    do_plot: bool = True,
    **metric_specific_kwargs,
):
    """Compute the given PPC metric for the given model, data and indices. Plot results by default"""
    # determine indices to use
    if custom_indices is not None:
        indices = custom_indices
    elif n_indices is not None:
        indices = np.random.randint(0, adata.n_obs, n_indices)
    else:
        indices = np.arange(adata.n_obs)

    sp, model_metrics, raw_metrics = _get_ppc_metrics(
        adata,
        model,
        metric,
        n_samples=n_samples,
        indices=indices,
        layer=layer,
        **metric_specific_kwargs,
    )

    if do_plot:
        model_name = f"{model.__class__.__name__}"
        plot_ppc(
            f"model={model_name} | metric={metric} | n_cells={len(indices)}",
            model_metrics,
            raw_metrics,
            metric,
            sp,
        )

    return sp
