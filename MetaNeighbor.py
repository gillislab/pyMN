import numpy as np
import pandas as pd
import bottleneck

import gc

import logging

from utilities import *


def MetaNeighbor(adata,
                 study_col,
                 ct_col,
                 genesets,
                 fast_version=False,
                 node_degree_normalization=True):
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

    shared_genes = adata.var_names[np.in1d(adata.var_names, genesets.index)]
    assert shared_genes.shape[
        0] > 1, 'No matching genes between genesets and sample matrix'

    genesets = genesets.loc[shared_genes]
    genesets = genesets.loc[:,genesets.sum()>0]
    
    assert genesets.shape[1] >0,'All Genesets are empty'

    adata_genes = adata.var_names
    results = {}
    for gset in genesets.columns:
        genes = genesets.index[genesets[gset].astype(bool)]
        adata_gs = adata[:, genes].X
        if fast_version:
            results[gset] = score_low_mem(adata_gs,
                                          adata.obs[study_col].values,
                                          adata.obs[ct_col].values,
                                          node_degree_normalization)
        else:
            results[gset] = score_default(adata_gs,
                                          adata.obs[study_col].values,
                                          adata.obs[ct_col].values,
                                          node_degree_normalization, means=True)

        del adata_gs
        gc.collect()
    return pd.DataFrame(results)


def score_low_mem(X, S, C, node_degree_normalization):
    pass


def score_default(X, S, C, node_degree_normalization, means=True):
    nw = create_nw_spearman(X.T)
    cell_labels = design_matrix(C)
    x1 = cell_labels.shape[1]
    x2 = cell_labels.shape[0]

    studies = np.unique(S)

    test_cell_labels = []
    for study in studies:
        nl = cell_labels.values.copy()
        nl[S == study, :] = 0
        test_cell_labels.append(nl.T)
    test_cell_labels = np.concatenate(test_cell_labels).T

    sum_in = nw @ test_cell_labels

    if node_degree_normalization:
        sum_all = np.sum(nw, axis=0)
        sum_in /= sum_all[:, None]

    sum_in[np.where(test_cell_labels==1)] = np.nan

    filter_mat = []
    for study in studies:
        nl = cell_labels.values.copy()
        nl[S != study, :] = np.nan
        filter_mat.append(nl.T)
    filter_mat = np.concatenate(filter_mat).T

    sum_in[np.isnan(filter_mat)] = np.nan

    sum_in = bottleneck.nanrankdata(np.abs(sum_in), axis=0)
    sum_in[filter_mat == 0] = 0

    n_p = bottleneck.nansum(filter_mat, axis=0)
    nn = filter_mat.shape[0] - n_p
    p = bottleneck.nansum(sum_in, axis=0)
    rocNV = (p / n_p - (n_p + 1) / 2) / nn

    #C array opposite of F in R
    rocNV = rocNV.reshape([studies.shape[0], x1]).T
    if means:
        return pd.Series(bottleneck.nanmean(rocNV, axis=1),
                         index=cell_labels.columns)
    else:
        return pd.DataFrame(rocNV, index=cell_labels.columns, columns=studies)
    return scores
