import numpy as np
import pandas as pd

from .utils import create_cell_labels, design_matrix


def topHits(
    adata,
    study_col=None,
    ct_col=None,
    cell_nv=None,
    mn_key="MetaNeighborUS",
    save_uns=True,
    threshold=0.95,
):
    """Find reciprocal top hits

    Identifies reciprocal top hits and hig scoring cell type pairs

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together

    Keyword Arguments:
        study_col {str or vector} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['study_col']
            else passed a vector (default: {None})
        ct_col {str or vector} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['ct_col']
            else passed a vector (default: {None})
        cell_nv {[type]} -- If None, inferrefed from adata.uns[f'{mn_key}'] else passed a pd.DataFrame (default: {None})
        mn_key {str} -- Location of MetaNeighborUS results (default: {'MetaNeighborUS'})
        save_uns {bool} --  Whether or not to save data in adata.uns[f'{mn_key}_topHits'] or return pd.DataFrame results (default: {True})
        threshold {number} -- AUROC threshold for calling a hit (default: {.95})
    """

    if cell_nv is not None:
        assert study_col in adata.obs_keys(), "Study Col not in adata"
        assert ct_col in adata.obs_keys(), "Cluster Col not in adata"
        cnv = cell_nv.copy()
    if cell_nv is None:
        assert (
            mn_key in adata.uns_keys()
        ), "MetaNeighborUS resutls not stored in adata or passed as cell_nv"
        cnv = adata.uns[mn_key].copy()
        study_col = adata.uns[f"{mn_key}_params"]["study_col"]
        ct_col = adata.uns[f"{mn_key}_params"]["ct_col"]

    assert np.all(
        cnv.index == cnv.columns
    ), "cell_nv is does not have the same order in both the rows and columns"

    pheno, _, study_ct_uniq = create_cell_labels(adata, study_col, ct_col)
    table_by_study = pheno["study_ct"].value_counts()

    pheno2 = pheno.drop_duplicates()
    pheno2.set_index("study_ct", inplace=True)

    # Set all AUROCS for self dataset and self to 0
    study_design = design_matrix(pheno2.loc[cnv.index, study_col].values)
    study_mask = study_design @ study_design.T
    #convert to boolean and numpy as it seems the duplicated col + row names result in the mask failing
    cnv.mask(study_mask.astype(bool).to_numpy(), other=0, inplace=True)
    np.fill_diagonal(cnv.values, 0)

    top_cols = pd.concat([cnv.max(axis=0), cnv.idxmax()], axis=1).reset_index()

    top_cols.columns = ["Study_ID|Celltype_1", "Mean_AUROC", "Study_ID|Celltype_2"]

    top_cols.loc[:, "Mean_AUROC"] = cnv.lookup(
        cnv.index, top_cols["Study_ID|Celltype_2"]
    )

    top_cols.loc[:, "Reciprocal"] = top_cols["Mean_AUROC"].duplicated()

    recip = top_cols[top_cols.Reciprocal]
    filt = top_cols.Mean_AUROC >= threshold
    res = pd.concat([recip, top_cols[filt]])
    res.loc[:, "Match_type"] = np.concatenate(
        [
            np.repeat("Reciprocal_top_hit", res.shape[0] - filt.sum()),
            np.repeat(f"Above_{threshold}", filt.sum()),
        ]
    )

    res = res.loc[~res.Mean_AUROC.duplicated()]
    res = res[
        ["Study_ID|Celltype_1", "Study_ID|Celltype_2", "Mean_AUROC", "Match_type"]
    ]
    res.sort_values("Mean_AUROC", ascending=False, inplace=True)
    res.reset_index(drop=True, inplace=True)
    res.Mean_AUROC = np.round(res.Mean_AUROC, 2)
    res = res.loc[res.Mean_AUROC >= threshold]
    if save_uns:
        adata.uns[f"{mn_key}_topHits"] = res
    else:
        return res
