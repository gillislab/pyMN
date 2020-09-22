import numpy as np
import pandas as pd

from .utils import create_cell_labels, design_matrix


def topHits(adata,
            study_col=None,
            ct_col=None,
            cell_nv=None,
            mn_key='MetaNeighborUS',
            threshold=.95):

    if cell_nv is not None:
        assert study_col in adata.obs_keys(), 'Study Col not in adata'
        assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'

    if cell_nv is None:
        assert mn_key in adata.uns_keys(
        ), 'MetaNeighborUS resutls not stored in adata or passed as cell_nv'
        cell_nv = adata.uns[mn_key].copy()
        study_col = adata.uns[f'mn_key_params']['study_col']
        ct_col = adata.uns[f'mn_key_params']['ct_col']

    assert np.all(
        cell_nv.index == cell_nv.columns
    ), 'cell_nv is does not have the same order in both the rows and columns'

    pheno, _, study_ct_uniq = create_cell_labels(adata, study_col, ct_col)
    table_by_study = pheno['study_ct'].value_counts()

    pheno2 = pheno.drop_duplicates()
    pheno2.set_index('study_ct', inplace=True)

    # Set all AUROCS for self dataset and self to 0
    study_design = design_matrix(pheno2.loc[cell_nv.index, study_col].values)
    study_mask = study_design @ study_design.T
    cell_nv.mask(study_mask.astype(bool), other=0, inplace=True)
    np.fill_diagonal(cell_nv.values, 0)

    top_cols = pd.concat(
        [cell_nv.max(axis=0), cell_nv.idxmax()], axis=1).reset_index()

    top_cols.columns = [
        'Study_ID|Celltype_1', 'Mean_AUROC', 'Study_ID|Celltype_2'
    ]

    top_cols['Mean_AUROC'] = cell_nv.lookup(cell_nv.index,
                                            top_cols['Study_ID|Celltype_2'])

    top_cols['Reciprocal'] = top_cols['Mean_AUROC'].duplicated()

    recip = top_cols[top_cols.Reciprocal]
    filt = top_cols.Mean_AUROC >= threshold
    res = pd.concat([recip, top_cols[filt]])
    res['Match_type'] = np.concatenate([
        np.repeat('Reciprocal_top_hit', res.shape[0] - filt.sum()),
        np.repeat(f'Above_{threshold}', filt.sum())
    ])

    res = res[~res.Mean_AUROC.duplicated()]
    res = res[[
        "Study_ID|Celltype_1", "Study_ID|Celltype_2", "Mean_AUROC",
        "Match_type"
    ]]
    res.sort_values('Mean_AUROC', ascending=False, inplace=True)
    res.reset_index(drop=True, inplace=True)
    res.Mean_AUROC = np.round(res.Mean_AUROC, 2)
    res = res[res.Mean_AUROC >= threshold]
    return res
