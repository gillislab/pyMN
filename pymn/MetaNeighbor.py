import numpy as np
import pandas as pd
import bottleneck

from scipy import sparse
import gc

from .utils import *


def MetaNeighbor(adata,
                 study_col,
                 ct_col,
                 genesets,
                 fast_version=False,
                 node_degree_normalization=True,
                 save_uns=True,
                 mn_key='MetaNeighbor'):
    assert study_col in adata.obs_keys(), 'Study Col not in adata'
    assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'
    assert ~isinstance(
        adata.obs[study_col].values[0],
        float), 'Study Col is a floating point, must be string or int'
    assert ~isinstance(
        adata.obs[ct_col].values[0],
        float), 'Cell Type Col is a floating point, must be string or int'
    assert np.unique(
        adata.obs[study_col]
    ).shape[0] > 1, f'Found only 1 unique study_id in {study_col}'

    shared_genes = np.intersect1d(adata.var_names.values,
                                  genesets.index.values)
    assert shared_genes.shape[
        0] > 1, 'No matching genes between genesets and sample matrix'

    genesets = genesets.loc[shared_genes]
    genesets = genesets.loc[:, genesets.sum() > 0]
    adata  = adata[:,shared_genes]

    assert genesets.shape[1] > 0, 'All Genesets are empty'
    genesets=genesets.astype(bool)
    results = {}

    study_vec = adata.obs[study_col].values 
    ct_vec = adata.obs[ct_col].values
    for gset in genesets.columns:
        adata_gs = adata.X[:,genesets[gset].values]
        if fast_version:
            results[gset] = score_low_mem(adata_gs,
                                          study_vec,
                                          ct_vec,
                                          node_degree_normalization)
        else:
            results[gset] = score_default(adata_gs,
                                          study_vec,
                                          ct_vec,
                                          node_degree_normalization,
                                          means=True)
    if save_uns:
        adata.uns[mn_key] = pd.DataFrame(results)
        adata.uns[f'{mn_key}_params']  = {
        'fast':fast_version,
        'node_degree_normalization': node_degree_normalization,
        'study_col':study_col,
        'ct_col':ct_col
        }
    else:
        return pd.DataFrame(results)


def score_low_mem(X, S, C, node_degree_normalization):
    slice_cells = np.ravel(np.sum(X, axis=1) > 0)
    X = X[slice_cells, :]
    S = S[slice_cells]
    C = C[slice_cells]
    
    cell_labels = design_matrix(C)
    cell_cols = cell_labels.columns
    cell_labels = cell_labels.values

    X_norm = np.asfortranarray(normalize_cells(X).T)
    studies = np.unique(S)

    res = {}
    for study in studies:
        votes = compute_votes(X_norm[:, S == study], X_norm[:, S != study],
                              cell_labels[S != study],
                              node_degree_normalization)
        votes = pd.DataFrame(votes,
                             index=C[S == study],
                             columns=cell_cols)
        roc = compute_aurocs(votes)
        res[study] = np.diag(roc.reindex(roc.columns).values)
    res = np.nanmean(pd.DataFrame(res), axis=1)
    res = pd.Series(res,cell_cols)
    return res


def compute_votes(candidates, voters, voter_id, node_degree_normalization):

    votes = np.dot(candidates.transpose(), np.dot(voters, voter_id))
    if node_degree_normalization:
        node_degree = np.sum(voter_id, axis=0)
        votes += node_degree
        norm = np.dot(
            candidates.transpose() , np.sum(voters, axis=1)) + voters.shape[1]
        votes = (votes.transpose() / norm).transpose()
    return votes


def score_default(X, S, C, node_degree_normalization, means=True):
    nw = create_nw_spearman(X.T)
    nw = (nw + nw.T) / 2

    cell_labels = design_matrix(C)

    x1 = cell_labels.shape[1]
    x2 = cell_labels.shape[0]

    studies = np.unique(S)
    exp_cols = np.repeat(studies, x1)

    test_cell_labels = np.tile(cell_labels.values, studies.shape[0])
    for study in studies:  #Hide testing labels
        d = np.where(study == S)[0]
        a = np.where(study == exp_cols)[0]
        for i in a:
            test_cell_labels[d, i] = 0


    predicts = nw @ test_cell_labels

    if node_degree_normalization:
        sum_all = np.sum(nw, axis=0)
        predicts /= sum_all[:, None]

    predicts[test_cell_labels == 1] = np.nan

    exp_cols = np.repeat(studies, x1)

    filter_mat = np.tile(cell_labels.values, studies.shape[0])
    for study in studies:
        mask = (study != S).astype(float)[:, None] @ (
            study == exp_cols).astype(float)[:, None].T
        mask = mask.astype(bool)
        filter_mat[mask] = np.nan
        predicts[mask] = np.nan

    predicts = bottleneck.nanrankdata(np.abs(predicts), axis=0)
    predicts[filter_mat == 0] = 0

    n_p = bottleneck.nansum(filter_mat, axis=0)
    n_n = bottleneck.nansum((filter_mat == 0).astype(float), axis=0)
    p = bottleneck.nansum(predicts, axis=0)
    rocNV = (p / n_p - (n_p + 1) / 2) / n_n

    #C array opposite of F in R
    rocNV = rocNV.reshape([studies.shape[0], x1]).T
    if means:
        return pd.Series(bottleneck.nanmean(rocNV, axis=1),
                         index=cell_labels.columns)
    else:
        return pd.DataFrame(rocNV, index=cell_labels.columns, columns=studies)
    return scores

