import pandas as pd
from scipy.sparse import coo_matrix
from scvi.data import synthetic_iid
from scvi.model import SCVI

from scvi_criticism import PPC


def test_ppc_init():
    adata = synthetic_iid()
    raw_counts = adata.X
    ppc = PPC(raw_counts=raw_counts, n_samples=42)

    assert isinstance(ppc.raw_counts, coo_matrix)
    assert ppc.posterior_predictive_samples == {}
    assert ppc.n_samples == 42
    assert ppc.models == {}
    assert ppc.metrics == {}


def get_ppc_with_samples(adata, two_models=True, n_samples=2):
    raw_counts = adata.X
    ppc = PPC(raw_counts=raw_counts, n_samples=n_samples)

    # create and train models
    SCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
    )
    model1 = SCVI(adata, n_latent=5)
    model1.train(1)

    if two_models:
        adata2 = adata.copy()
        SCVI.setup_anndata(
            adata2,
            batch_key="batch",
        )
        model2 = SCVI(adata2, n_latent=5)
        model2.train(1)

    models_dict = {"model1": model1}
    if two_models:
        models_dict["model2"] = model2

    ppc.store_posterior_predictive_samples(models_dict)

    return ppc, models_dict


def test_ppc_get_samples():
    adata = synthetic_iid()
    ppc, models_dict = get_ppc_with_samples(adata)

    assert ppc.models == models_dict
    assert len(ppc.posterior_predictive_samples.keys()) == 2
    # n_cells x n_genes x n_samples
    assert ppc.posterior_predictive_samples["model1"].shape == (400, 100, 2)
    assert ppc.posterior_predictive_samples["model2"].shape == (400, 100, 2)


def test_ppc_cv_mwu():
    adata = synthetic_iid(n_genes=10)
    ppc, _ = get_ppc_with_samples(adata)

    ppc.coefficient_of_variation(cell_wise=True)
    ppc.coefficient_of_variation(cell_wise=False)
    ppc.mann_whitney_u()

    assert list(ppc.metrics.keys()) == ["cv_cell", "cv_gene", "mannwhitneyu"]

    assert isinstance(ppc.metrics["cv_cell"], pd.DataFrame)
    assert ppc.metrics["cv_cell"].columns.tolist() == ["model1", "model2", "Raw"]
    assert ppc.metrics["cv_cell"].index.equals(pd.RangeIndex(start=0, stop=400))

    assert isinstance(ppc.metrics["cv_gene"], pd.DataFrame)
    assert ppc.metrics["cv_gene"].columns.tolist() == ["model1", "model2", "Raw"]
    assert ppc.metrics["cv_gene"].index.equals(pd.RangeIndex(start=0, stop=10))

    assert isinstance(ppc.metrics["mannwhitneyu"], pd.DataFrame)
    assert ppc.metrics["mannwhitneyu"].columns.tolist() == ["model1", "model2"]
    assert ppc.metrics["mannwhitneyu"].index.equals(pd.RangeIndex(start=0, stop=10))


def test_ppc_de():
    adata = synthetic_iid()
    adata.var["my_gene_names"] = [f"gene_{i}" for i in range(adata.shape[1])]
    ppc, _ = get_ppc_with_samples(adata, two_models=False, n_samples=1)

    ppc.diff_exp(
        adata.obs,
        adata.var,
        de_groupby="labels",
        var_gene_names_col="my_gene_names",
        n_top_genes=2,
        n_top_genes_overlap=5,
    )

    assert list(ppc.metrics.keys()) == ["diff_exp"]

    assert isinstance(ppc.metrics["diff_exp"], dict)
    assert set(ppc.metrics["diff_exp"].keys()) == {"adata_raw", "var_names", "model1", "lfc_df_raw", "fraction_df_raw"}

    de_adata_raw = ppc.metrics["diff_exp"]["adata_raw"]
    assert de_adata_raw.shape == adata.shape

    de_var_names = ppc.metrics["diff_exp"]["var_names"]
    labels_set = set(adata.obs["labels"].unique())
    assert set(de_var_names.keys()) == labels_set
    for val in de_var_names.values():
        assert len(val) == 2  # n_top_genes
    # flatten all n_labels x n_top_genes into a single set
    genes_set = {e for l in de_var_names.values() for e in l}

    de_lfc_df_raw = ppc.metrics["diff_exp"]["lfc_df_raw"]
    assert set(de_lfc_df_raw.index) == labels_set
    assert set(de_lfc_df_raw.columns) == genes_set

    de_fraction_df_raw = ppc.metrics["diff_exp"]["fraction_df_raw"]
    assert set(de_fraction_df_raw.index) == labels_set
    assert set(de_fraction_df_raw.columns) == genes_set

    # validate model1 results
    de_model1 = ppc.metrics["diff_exp"]["model1"]
    assert isinstance(de_model1, dict)
    assert set(de_model1.keys()) == {
        "adata_approx",
        "lfc_df_approx",
        "lfc_mae",
        "lfc_mae_mean",
        "lfc_pearson",
        "lfc_pearson_mean",
        "lfc_spearman",
        "lfc_spearman_mean",
        "fraction_df_approx",
        "fraction_mae",
        "fraction_mae_mean",
        "fraction_pearson",
        "fraction_pearson_mean",
        "fraction_spearman",
        "fraction_spearman_mean",
        "gene_comparisons",
    }

    de_model1_adata_approx = de_model1["adata_approx"]
    assert de_model1_adata_approx.shape == adata.shape

    for kind in ["lfc", "fraction"]:
        de_model1_df_approx = de_model1[f"{kind}_df_approx"]
        assert set(de_model1_df_approx.index) == labels_set
        assert set(de_model1_df_approx.columns) == genes_set

        for metric in ["mae", "pearson", "spearman"]:
            de_model1_metric = de_model1[f"{kind}_{metric}"]
            assert isinstance(de_model1_metric, pd.Series)
            assert set(de_model1_metric.index) == labels_set
            assert de_model1[f"{kind}_{metric}_mean"].ndim == 0

    de_model1_gene_comparisons = de_model1["gene_comparisons"]
    assert isinstance(de_model1_gene_comparisons, pd.DataFrame)
    assert set(de_model1_gene_comparisons.index) == labels_set
    assert set(de_model1_gene_comparisons.columns) == {"precision", "recall", "f1"}
