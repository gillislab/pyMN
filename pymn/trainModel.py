from .utils import *


def trainModel(
    adata, study_col, ct_col, var_genes="highly_variable"):
    """Pretrains model for the unsupervised version of MetaNeighbor

    When comparing clusters to a large reference dataset, this function
    summarizes the gene-by-cell matrix into a much smaller highly variable
    gene-by-cluster matrix which can be fed as training data into MetaNeighborUS,
    resulting in substantial time and memory savings.

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
        ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels

    Keyword Arguments:
        var_genes {str or vector} -- String that identifies boolean vector of highly variable genes
            in adata.var or vector of highly variable genes (default: {'highly_variable'})

    Returns:
        pd.DataFrame -- DataFrame containing trained model, variable genes x cell types.
            Needed for running MetaNeighborUS with a pretrained model
    """
    assert study_col in adata.obs_keys(), "Study Col not in adata"
    assert ct_col in adata.obs_keys(), "Cluster Col not in adata"

    if var_genes is not "highly_variable":
        var_genes = adata.var_names[np.in1d(adata.var_names, var_genes)]
    else:
        var_genes = adata.var_names[adata.var[var_genes]]
    assert var_genes.shape[0] > 2, "Must have at least 2 genes"

    adata = adata[:, var_genes]

    dat = normalize_cells(adata.X).T
    labels = join_labels(adata.obs[study_col].values, adata.obs[ct_col].values)
    label_matrix = design_matrix(labels)
    result = pd.DataFrame(
        dat @ label_matrix.values, index=var_genes, columns=label_matrix.columns
    )
    n_cells = label_matrix.sum()
    n_cells.name = "n_cells"
    n_cells = pd.DataFrame(n_cells)
    return pd.concat([n_cells.T, result])
