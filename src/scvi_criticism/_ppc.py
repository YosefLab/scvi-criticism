import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import pearsonr
from scvi.model.base import BaseModelClass
from sklearn.metrics import average_precision_score, roc_auc_score
from sparse import GCXS, SparseArray
from xarray import DataArray, Dataset

from ._constants import (
    DATA_VAR_RAW,
    DEFAULT_DE_N_TOP_GENES_OVERLAP,
    DEFAULT_P_VAL_THRESHOLD,
    METRIC_CALIBRATION,
    METRIC_CV_CELL,
    METRIC_CV_GENE,
    METRIC_DIFF_EXP,
    METRIC_ZERO_FRACTION,
    UNS_NAME_RGG_PPC,
    UNS_NAME_RGG_RAW,
)

Dims = Literal["cells", "features"]


def _make_dataset_dense(dataset: Dataset) -> Dataset:
    """Make a dataset dense, converting sparse arrays to dense arrays."""
    dataset = dataset.map(lambda x: x.data.todense() if isinstance(x.data, SparseArray) else x)
    return dataset


@dataclass
class MetricConfig:
    """
    Metric config for running posterior predictive checks.

    Attributes
    ----------
    method_name
        Name of the method to run. Must be a method of :class:`~scvi_criticism.PosteriorPredictiveCheck`.
    method_kwargs
        Keyword arguments to pass to the method
    """

    method_name: str
    method_kwargs: Dict[str, Any]


class PosteriorPredictiveCheck:
    """
    Posterior predictive checks for comparing single-cell generative models

    Parameters
    ----------
    adata
        AnnData object with raw counts.
    models_dict
        Dictionary of models to compare.
    count_layer_key
        Key in adata.layers to use as raw counts, if None, use adata.X.
    n_samples
        Number of posterior predictive samples to generate
    """

    def __init__(
        self,
        adata: AnnData,
        models_dict: Dict[str, BaseModelClass],
        count_layer_key: Optional[str] = None,
        n_samples: int = 10,
    ):
        self.adata = adata
        self.count_layer_key = count_layer_key
        raw_counts = adata.layers[count_layer_key] if count_layer_key is not None else adata.X
        if isinstance(raw_counts, np.ndarray):
            self.raw_counts = GCXS.from_numpy(raw_counts)
        elif issparse(raw_counts):
            self.raw_counts = GCXS.from_scipy_sparse(raw_counts)
        else:
            raise ValueError("raw_counts must be a numpy array or scipy sparse matrix")
        self.samples_dataset = None
        self.n_samples = n_samples
        self.models = models_dict
        self.metrics = {}

        self._store_posterior_predictive_samples()

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

    def _store_posterior_predictive_samples(
        self,
        batch_size: int = 32,
        indices: List[int] = None,
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
        self.batch_size = batch_size

        samples_dict = {}
        for m, model in self.models.items():
            pp_counts = model.posterior_predictive_sample(
                self.adata,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                indices=indices,
            )
            samples_dict[m] = DataArray(
                data=pp_counts,
                coords={
                    "cells": self.adata.obs_names,
                    "features": model.adata.var_names,
                    "samples": np.arange(self.n_samples),
                },
            )
        samples_dict[DATA_VAR_RAW] = DataArray(
            data=self.raw_counts, coords={"cells": self.adata.obs_names, "features": self.adata.var_names}
        )
        self.samples_dataset = Dataset(samples_dict)

    def coefficient_of_variation(self, dim: Dims = "cells") -> None:
        """
        Calculate the coefficient of variation (CV) for each model and the raw counts.

        The CV is computed over the cells or features dimension per sample. The mean CV is then
        computed over all samples.

        Parameters
        ----------
        dim
            Dimension to compute CV over.
        """
        identifier = METRIC_CV_CELL if dim == "features" else METRIC_CV_GENE
        pp_samples = self.samples_dataset
        std = pp_samples.std(dim=dim, skipna=False)
        mean = pp_samples.mean(dim=dim, skipna=False)
        cv = std / mean
        # It's ok to make things dense here
        cv = _make_dataset_dense(cv)
        cv_mean = cv.mean(dim="samples", skipna=True)
        cv_mean[DATA_VAR_RAW].data = np.nan_to_num(cv_mean[DATA_VAR_RAW].data)
        self.metrics[identifier] = cv_mean.to_dataframe()

    def zero_fraction(self) -> None:
        """Fraction of zeros in raw counts for a specific gene"""
        pp_samples = self.samples_dataset
        mean = pp_samples.mean(dim="cells", skipna=False).mean(dim="samples", skipna=False)
        mean = _make_dataset_dense(mean)
        self.metrics[METRIC_ZERO_FRACTION] = mean.to_dataframe()

    def calibration_error(self, confidence_intervals: Optional[List[float]] = None) -> None:
        """Calibration error for each observed count.

        For a series of credible intervals of the samples, the fraction of observed counts that fall
        within the credible interval is computed. The calibration error is then the squared difference
        between the observed fraction and the true interval width.

        For this metric, lower is better.

        Parameters
        ----------
        confidence_intervals
            List of confidence intervals to compute calibration error for.
            E.g., [0.01, 0.02, 0.98, 0.99]
        """
        if confidence_intervals is None:
            ps = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 82.5, 85, 87.5, 90, 92.5, 95, 97.5]
            ps = [p / 100 for p in ps]
        else:
            if len(confidence_intervals) % 2 != 0:
                raise ValueError("Confidence intervals must be even")
            ps = confidence_intervals
        pp_samples = self.samples_dataset
        # results in (quantiles, cells, features)
        quants = pp_samples.quantile(q=ps, dim="samples", skipna=False)
        credible_interval_indices = [(i, len(ps) - (i + 1)) for i in range(len(ps) // 2)]

        model_cal = {}
        for model in pp_samples.data_vars:
            if model == DATA_VAR_RAW:
                continue
            cal_error_features = 0
            for interval in credible_interval_indices:
                start = interval[0]
                end = interval[1]
                true_width = ps[end] - ps[start]
                greater_than = (quants[DATA_VAR_RAW] >= quants.model1.isel(quantile=start)).data
                less_than = (quants[DATA_VAR_RAW] <= quants.model1.isel(quantile=end)).data
                # Logical and
                ci = greater_than * less_than
                pci_features = ci.mean()
                cal_error_features += (pci_features - true_width) ** 2
            model_cal[model] = {
                "features": cal_error_features,
            }
        self.metrics[METRIC_CALIBRATION] = pd.DataFrame.from_dict(model_cal)

    def differential_expression(
        self,
        de_groupby: str,
        de_method: str = "t-test",
        n_samples: int = 1,
        cell_scale_factor: float = 1e4,
        p_val_thresh: float = DEFAULT_P_VAL_THRESHOLD,
        n_top_genes_fallback: int = DEFAULT_DE_N_TOP_GENES_OVERLAP,
    ):
        """
        Compute differential expression (DE) metrics.

        If n_samples > 1, all metrics are averaged over a posterior predictive dataset.

        Parameters
        ----------
        de_groupby
            The column name in `adata_obs_raw` that contains the groupby information.
        de_method
            The DE method to use. See :meth:`~scanpy.tl.rank_genes_groups` for more details.
        n_samples
            The number of posterior predictive samples to use for the DE analysis.
        cell_scale_factor
            The cell scale factor to use for normalization before DE.
        p_val_thresh
            The p-value threshold to use for the DE analysis.
        n_top_genes_fallback
            The number of top genes to use for the DE analysis if the number of genes
            with a p-value < p_val_thresh is zero.
        """
        if n_samples > self.n_samples:
            raise ValueError(
                f"n_samples={n_samples} is greater than the number of samples already recorded " f"({self.n_samples})"
            )
        # run DE with the raw counts
        adata_de = AnnData(X=self.raw_counts.to_scipy_sparse().tocsr().copy(), obs=self.adata.obs, var=self.adata.var)
        sc.pp.normalize_total(adata_de, target_sum=cell_scale_factor)
        sc.pp.log1p(adata_de)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            sc.tl.rank_genes_groups(adata_de, de_groupby, use_raw=False, method=de_method, key_added=UNS_NAME_RGG_RAW)

        # get posterior predictive samples from the model (aka approx counts)
        pp_samples = self.samples_dataset
        # create adata object to run DE on the approx counts
        # X here will be overwritten
        adata_approx = AnnData(X=adata_de.X, obs=adata_de.obs, var=adata_de.var)
        de_keys = {}
        models = [model for model in pp_samples.data_vars if model != DATA_VAR_RAW]
        for model in models:
            if model not in de_keys:
                de_keys[model] = []
            for k in range(n_samples):
                one_sample = pp_samples[model].isel(samples=k)
                # overwrite X with the posterior predictive sample
                # This allows us to save all the DE results in the same adata object
                one_sample_data = (
                    one_sample.to_scipy_sparse().tocsr() if isinstance(one_sample, SparseArray) else one_sample
                )
                adata_approx.X = one_sample_data.copy()
                sc.pp.normalize_total(adata_approx, target_sum=cell_scale_factor)
                sc.pp.log1p(adata_approx)

                # run DE with the imputed normalized data
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
                    key_added = f"{UNS_NAME_RGG_PPC}_{model}_{k}"
                    de_keys[model].append(key_added)
                    sc.tl.rank_genes_groups(
                        adata_approx, de_groupby, use_raw=False, method=de_method, key_added=key_added
                    )

        groups = self.adata.obs[de_groupby].astype("category").cat.categories
        df = pd.DataFrame(
            index=np.arange(len(groups) * len(models)),
            columns=["roc_auc", "pr_auc", "lfc_mae", "lfc_pearson", "group", "model"],
        )
        i = 0
        for g in groups:
            raw_group_data = sc.get.rank_genes_groups_df(adata_de, group=g, key=UNS_NAME_RGG_RAW)
            raw_group_data.set_index("names", inplace=True)
            for model in de_keys.keys():
                roc_aucs = []
                pr_aucs = []
                lfc_maes = []
                lfc_pearsons = []
                # Now over potential samples
                for de_key in de_keys[model]:
                    sample_group_data = sc.get.rank_genes_groups_df(adata_approx, group=g, key=de_key)
                    sample_group_data.set_index("names", inplace=True)
                    sample_group_data = sample_group_data.loc[raw_group_data.index]
                    raw_adj_p_vals = raw_group_data["pvals_adj"]
                    true = raw_adj_p_vals < p_val_thresh
                    pred = sample_group_data["scores"]
                    if true.sum() == 0:
                        # if there are no true DE genes, just use the top n genes
                        true = np.zeros_like(pred)
                        true[np.argsort(raw_adj_p_vals)[:n_top_genes_fallback]] = 1
                    roc_aucs.append(roc_auc_score(true, pred))
                    pr_aucs.append(average_precision_score(true, pred))
                    lfc_maes.append(
                        np.mean(np.abs(raw_group_data["logfoldchanges"] - sample_group_data["logfoldchanges"]))
                    )
                    lfc_pearsons.append(
                        pearsonr(raw_group_data["logfoldchanges"], sample_group_data["logfoldchanges"])[0]
                    )
                # Mean here is over sampled datasets
                df.loc[i, "roc_auc"] = np.mean(roc_aucs)
                df.loc[i, "pr_auc"] = np.mean(pr_aucs)
                df.loc[i, "lfc_mae"] = np.mean(lfc_maes)
                df.loc[i, "lfc_pearson"] = np.mean(lfc_pearsons)
                df.loc[i, "model"] = model
                df.loc[i, "group"] = g
                i += 1

        self.metrics[METRIC_DIFF_EXP] = df

    def run(self, metrics_to_run: List[MetricConfig]):
        """Run the metrics."""
        # calculate metrics
        for metric_config in metrics_to_run:
            metric_specific_kwargs = metric_config.kwargs
            method_name = metric_config.method_name
            fn = getattr(self, method_name)
            fn(**metric_specific_kwargs)
