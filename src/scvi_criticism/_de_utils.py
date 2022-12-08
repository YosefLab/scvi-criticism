from typing import Optional

import pandas as pd
from anndata import AnnData

UNS_NAME_RGG = "rank_genes_groups"


def _get_top_n_genes_per_group(adata: AnnData, n_genes: int, var_gene_names_col: Optional[str] = None):
    rgg = adata.uns[UNS_NAME_RGG]
    rgg_names = pd.DataFrame.from_records(rgg["names"])
    group_to_genes = {}
    for group in rgg_names.columns:
        top_n_gene_ids = rgg_names[group].values[:n_genes]
        if var_gene_names_col is None:
            top_n_gene_names = top_n_gene_ids.tolist()
        else:
            top_n_gene_names = adata.var.loc[top_n_gene_ids][var_gene_names_col].values.tolist()
        group_to_genes[group] = top_n_gene_names
    return group_to_genes


def _get_all_de_groups(adata: AnnData):
    return adata.uns[UNS_NAME_RGG]["names"].dtype.names
