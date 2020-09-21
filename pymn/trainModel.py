from .utils import *


def train_model(adata, study_col, ct_col, var_genes='highly_variable'):
    assert study_col in adata.obs_keys(), 'Study Col not in adata'
    assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'

    if var_genes is not 'highly_variable':
        var_genes = adata.var_names[np.in1d(adata.var_names, var_genes)]
    else:
        var_genes = adata.var_names[adata.var[var_genes]]
    assert var_genes.shape[0] > 2, 'Must have at least 2 genes'

    adata = adata[:, var_genes]

    dat = normalize_cells(adata.X).T
    labels = join_labels(adata.obs[study_col].values, adata.obs[ct_col].values)
    label_matrix = design_matrix(labels)
    result = pd.DataFrame(dat @ label_matrix,
                          index=var_genes,
                          columns=label_matrix.columns)

    n_cells = pd.Series(bottleneck.nansum(label_matrix, axis=0),
                        index=label_matrix.columns,
                        name='n_cells')

    return pd.concat([result, n_cells])
