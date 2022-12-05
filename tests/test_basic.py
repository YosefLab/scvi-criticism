def test_criticism():
    pass
    # TODO update
    # adata = scvi.data.synthetic_iid()
    # adata.obs["size_factor"] = np.random.randint(1, 5, size=(adata.shape[0],))
    # scvi.model.SCVI.setup_anndata(
    #     adata,
    #     batch_key="batch",
    #     size_factor_key="size_factor",
    # )
    # adata_c = adata.copy()

    # model = scvi.model.SCVI(adata, n_latent=5)
    # model.train(1, check_val_every_n_epoch=1, train_size=0.5)

    # (
    #     adata.obsm["X_latent_qzm"],
    #     adata.obsm["X_latent_qzv"],
    # ) = model.get_latent_representation(give_mean=False, return_dist=True)
    # model.to_latent_mode(mode="dist")

    # # should fail if we dont give the full counts
    # with pytest.raises(ValueError):
    #     run_ppc(adata, model, "cv_cell", do_plot=False)

    # sp_train, sp_val = run_ppc(
    #     adata_c, model, "cv_cell", n_samples_train=2, n_samples_val=5, do_plot=False
    # )
    # assert sp_train is not None
    # train_indices_len = len(model.train_indices)
    # gene_counts = len(adata.var)
    # assert list(sp_train.metrics.keys()) == ["cv_cell"]
    # assert list(sp_train.models.keys()) == ["SCVI"]
    # assert len(sp_train.metrics["cv_cell"].index) == train_indices_len
    # assert sp_train.metrics["cv_cell"].columns.values.tolist() == ["SCVI", "Raw"]
    # assert sp_train.posterior_predictive_samples["SCVI"].shape == (
    #     train_indices_len,
    #     gene_counts,
    #     2,
    # )  # 2 is n_samples_train

    # assert sp_val is not None
    # val_indices_len = len(model.validation_indices)
    # assert list(sp_val.metrics.keys()) == ["cv_cell"]
    # assert list(sp_val.models.keys()) == ["SCVI"]
    # assert len(sp_val.metrics["cv_cell"].index) == val_indices_len
    # assert sp_val.metrics["cv_cell"].columns.values.tolist() == ["SCVI", "Raw"]
    # assert sp_val.posterior_predictive_samples["SCVI"].shape == (
    #     val_indices_len,
    #     gene_counts,
    #     5,
    # )  # 5 is n_samples_val

    # sp_train, sp_val = run_ppc(
    #     adata_c, model, "cv_gene", n_samples_train=2, n_samples_val=5, do_plot=False
    # )
    # assert sp_train is not None
    # assert sp_val is not None
    # assert list(sp_train.metrics.keys()) == ["cv_gene"]
    # assert list(sp_train.models.keys()) == ["SCVI"]
    # assert len(sp_train.metrics["cv_gene"].index) == gene_counts
    # assert sp_train.metrics["cv_gene"].columns.values.tolist() == ["SCVI", "Raw"]
    # assert sp_train.posterior_predictive_samples["SCVI"].shape == (
    #     train_indices_len,
    #     gene_counts,
    #     2,
    # )  # 2 is n_samples_train

    # sp_train, sp_val = run_ppc(
    #     adata_c,
    #     model,
    #     "mannwhitneyu",
    #     n_samples_train=2,
    #     n_samples_val=0,
    #     do_plot=False,
    # )
    # assert sp_train is not None
    # assert sp_val is None
    # assert list(sp_train.metrics.keys()) == ["mannwhitneyu"]
    # assert list(sp_train.models.keys()) == ["SCVI"]
    # assert len(sp_train.metrics["mannwhitneyu"].index) == gene_counts
    # assert sp_train.metrics["mannwhitneyu"].columns.values.tolist() == ["SCVI"]
    # assert sp_train.posterior_predictive_samples["SCVI"].shape == (
    #     train_indices_len,
    #     gene_counts,
    #     2,
    # )  # 2 is n_samples_train
