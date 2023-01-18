import json
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import mannwhitneyu

from ._constants import (
    DEFAULT_DE_N_TOP_GENES,
    DEFAULT_DE_N_TOP_GENES_OVERLAP,
    METRIC_CV_CELL,
    METRIC_CV_GENE,
    METRIC_DIFF_EXP,
    METRIC_MWU,
)
from ._de_utils import _get_all_de_groups, _get_top_n_genes_per_group
from ._utils import (
    _get_binary_array_from_selection,
    _get_df_corr_coeff,
    _get_df_mae,
    _get_dp_as_df,
    _get_precision_recall_f1,
)


class PPC:
    """
    Posterior predictive checks for comparing single-cell generative models

    Parameters
    ----------
    n_samples
        Number of posterior predictive samples to generate
    raw_counts
        Raw counts matrix (cells x genes) as a numpy array, scipy coo_matrix, or scipy csr_matrix
    """

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
                return f"AnnData object with n_obs={o.n_obs}, n_vars={o.n_vars}"
            elif isinstance(o, pd.DataFrame):
                s = f"Pandas DataFrame with shape={o.shape}, "
                n_cols = 5
                if len(o.columns) > n_cols:
                    return s + f"first {n_cols} columns={o.columns[:n_cols].to_list()}"
                return s + f"columns={o.columns.to_list()}"
            elif isinstance(o, pd.Series):
                return f"Pandas Series with n_rows={len(o)}"
            return f"ERROR unserializable type: {type(o)}"

        return json.dumps(self.metrics, indent=4, default=custom_handle_unserializable)

    def store_posterior_predictive_samples(
        self,
        models_dict,
        batch_size=32,
        indices=None,
    ):
        """
        Store posterior predictive samples for each model.

        Parameters
        ----------
        models_dict
            Dictionary of models to store posterior predictive samples for.
        batch_size
            Batch size for generating posterior predictive samples.
        indices
            Indices to generate posterior predictive samples for.
        """
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
        Calculate the coefficient of variation (CV) for each model and the raw counts.

        Parameters
        ----------
        cell_wise
            Whether to calculate the CV cell-wise or gene-wise.
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

    def mann_whitney_u(self):
        """Calculate the Mann-Whitney U test between each model and the raw counts."""
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

    def _diff_exp_compare_dotplots(self, rgg_dp_raw, rgg_dp_approx, m, kind):
        # compare the dotplots in terms of lfc/fraction values depending on `kind`

        assert kind in ["lfc", "fraction"]
        dp_kind = "color" if kind == "lfc" else "fraction"
        df_raw = _get_dp_as_df(rgg_dp_raw, dp_kind)
        df_approx = _get_dp_as_df(rgg_dp_approx, dp_kind)
        self.metrics[METRIC_DIFF_EXP][f"{kind}_df_raw"] = df_raw
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
        """
        Compute differential expression (DE) metrics.

        Parameters
        ----------
        adata_obs_raw
            The `obs` dataframe from the raw AnnData object.
        adata_var_raw
            The `var` dataframe from the raw AnnData object.
        de_groupby
            The column name in `adata_obs_raw` that contains the groupby information.
        de_method
            The DE method to use. See :meth:`~scanpy.tl.rank_genes_groups` for more details.
        var_gene_names_col
            The column name in `adata_var_raw` that contains the gene names. If `None`, then
            `adata_var_raw.index` is used.
        n_top_genes
            The number of top genes to use for the DE analysis. If `None`, then the default value
            `DEFAULT_DE_N_TOP_GENES` is used.
        n_top_genes_overlap
            The number of top genes to use for the DE analysis when computing the gene overlap
            metrics. If `None`, then the default value `DEFAULT_DE_N_TOP_GENES_OVERLAP` is used.
        """
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
            if samples.ndim == 3:
                raise NotImplementedError(
                    f"Currently only n_samples=1 is supported for DE.\
                    Got n_samples={samples.ndim}"
                )
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


def run_ppc(
    adata: AnnData,
    model,
    metric: str,
    n_samples: int,
    layer: Optional[str] = None,
    custom_indices: Optional[Union[int, Sequence[int]]] = None,
    **metric_specific_kwargs,
):
    """
    Compute the given PPC metric for the given model.

    Parameters
    ----------
    adata
        The raw AnnData object.
    model
        The model to compute the PPC metric for.
    metric
        The PPC metric to compute.
    n_samples
        The number of posterior predictive samples to draw.
    layer
        The layer in `adata` where the raw counts reside, if different from `adata.X`.
    custom_indices
        The indices to use for the PPC metric computation. If it is a list, we will use these indices. Else if it is
        an integer, we will randomly draw those many indices from 0..adata.n_obs. Else if it is None, all indices are
        used.
    metric_specific_kwargs
        Keyword arguments for the metric-specific calls.

    Returns
    -------
    An instance of the :class:`~scvi_criticism.PPC` class that contains the computed metric results as an attribute.
    """
    # determine indices to use
    if isinstance(custom_indices, list):
        indices = custom_indices
    elif isinstance(custom_indices, int):
        indices = np.random.randint(0, adata.n_obs, custom_indices)
    else:
        indices = np.arange(adata.n_obs)

    # create PPC instance and compute pp samples
    raw_data = adata[indices].X if layer is None else adata[indices].layers[layer]
    ppc = PPC(n_samples=n_samples, raw_counts=raw_data)
    model_name = f"{model.__class__.__name__}"
    models_dict = {model_name: model}
    ppc.store_posterior_predictive_samples(models_dict, indices=indices)

    # calculate metrics
    if (metric == METRIC_CV_CELL) or (metric == METRIC_CV_GENE):
        ppc.coefficient_of_variation(cell_wise=(metric == METRIC_CV_CELL))
    elif metric == METRIC_MWU:
        ppc.mann_whitney_u()
    elif metric == METRIC_DIFF_EXP:
        # adata.obs is needed for de_groupby
        ppc.diff_exp(adata[indices].obs, adata.var, **metric_specific_kwargs)
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

    return ppc
