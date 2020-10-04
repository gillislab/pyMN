import numpy as np
import pandas as pd
import bottleneck

from scipy import sparse
import gc

from .utils import *


def MetaNeighbor(
    adata,
    study_col,
    ct_col,
    genesets,
    node_degree_normalization=True,
    save_uns=True,
    fast_version=False,
    fast_hi_mem=False,
    mn_key="MetaNeighbor",
):
    """Runs MetaNeighbor

     For each gene set of interest, the function builds a network of rank
     correlations between all cells. Next,It builds a network of rank correlations
     between all cells for a gene set. Next, the neighbor voting predictor
     produces a weighted matrix of predicted labels by performing matrix
     multiplication between the network and the binary vector indicating cell type
     membership, then dividing each element by the null predictor (i.e., node
     degree). That is, each cell is given a score equal to the fraction of its
     neighbors (including itself), which are part of a given cell type. For
     cross-validation, we permute through all possible combinations of
     leave-one-dataset-out cross-validation, and we report how well we can recover
     cells of the same type as area under the receiver operator characteristic
     curve (AUROC). This is repeated for all folds of cross-validation, and the
     mean AUROC across folds is reported.

    Arguments:
        adata {AnnData} -- Object containing all single cell experiements conactenated
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
        ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels
        genesets {pd.DataFrame} -- One hot encoded dataframe of genes x gene sets

    Keyword Arguments:
        node_degree_normalization {bool} -- Flag for normalizing votes by node degree (default: {True})
        save_uns {bool} -- Flag for saving results in adata.uns[mn_key], return if False (default: {True})
        fast_version {bool} -- Flag for low memory fast version (default: {False})
        fast_hi_mem {bool} -- Flag for slightly faster  (default: {False})
        mn_key {str} -- String for storing results in adata.uns (default: {'MetaNeighbor'})

    Returns:
        None/pd.DataFrame -- if save_uns is False, return dataframe of cell-type x gene set AUROCs
    """
    assert study_col in adata.obs_keys(), "Study Col not in adata"
    assert ct_col in adata.obs_keys(), "Cluster Col not in adata"
    assert ~isinstance(
        adata.obs[study_col].values[0], float
    ), "Study Col is a floating point, must be string or int"
    assert ~isinstance(
        adata.obs[ct_col].values[0], float
    ), "Cell Type Col is a floating point, must be string or int"
    assert (
        np.unique(adata.obs[study_col]).shape[0] > 1
    ), f"Found only 1 unique study_id in {study_col}"

    shared_genes = np.intersect1d(adata.var_names.values, genesets.index.values)
    assert (
        shared_genes.shape[0] > 1
    ), "No matching genes between genesets and sample matrix"

    genesets = genesets.loc[shared_genes]
    genesets = genesets.loc[:, genesets.sum() > 0]

    assert genesets.shape[1] > 0, "All Genesets are empty"
    genesets = genesets.astype(bool)
    results = {}

    study_vec = adata.obs[study_col].values
    ct_vec = adata.obs[ct_col].values
    if fast_hi_mem:  # Stores as dense arrray (faster)
        expression = adata[:, shared_genes].X.toarray()
    else:
        expression = adata[:, shared_genes].X
    for gset in genesets.columns:
        adata_gs = expression[:, np.where(genesets[gset].values)[0]]
        if fast_version:
            results[gset] = score_low_mem(
                adata_gs, study_vec, ct_vec, node_degree_normalization
            )
        else:
            results[gset] = score_default(
                adata_gs, study_vec, ct_vec, node_degree_normalization, means=True
            )
    if save_uns:
        adata.uns[mn_key] = pd.DataFrame(results)
        adata.uns[f"{mn_key}_params"] = {
            "fast": fast_version,
            "node_degree_normalization": node_degree_normalization,
            "study_col": study_col,
            "ct_col": ct_col,
        }
    else:
        return pd.DataFrame(results)


def score_low_mem(
    X, S, C, node_degree_normalization):
    """Compute Neighbor Voting using low memory method

    Compute using the approximate low memory method

    Arguments:
        X {array} -- Array (sparse or dense) of geneset x cells
        S {vector} -- Study labels, length cells
        C {vector} -- Cell type labels, legnth cells
        node_degree_normalization {bool} -- Flag for whether to normalize votes by node degree

    Returns:
        pd.Series -- Series containing AUROCs for each cell type for the given gene set
    """
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
        is_study = np.where(S == study)[0]
        is_not_study = np.where(S != study)[0]
        votes = compute_votes(
            X_norm[:, is_study].T,
            X_norm[:, is_not_study],
            cell_labels[is_not_study],
            node_degree_normalization,
        )
        votes = pd.DataFrame(votes, index=C[is_study], columns=cell_cols)
        roc = compute_aurocs(votes)
        res[study] = np.diag(roc.reindex(roc.columns).values)
    res = np.nanmean(pd.DataFrame(res), axis=1)
    res = pd.Series(res, index=cell_cols)
    return res


def compute_votes(
    candidates,
    voters,
    voter_id,
    node_degree_normalization,
):
    """Comptue neighbor voting for a given set of candidates and voters


    Arguments:
        candidates {np.ndarray} -- genes x cells normalized expression for candidates
        voters {np.ndarray} -- genes x cells normalized expression for voters
        voter_id {np.ndarray} -- design_matrix for voters for cell type identities
        node_degree_normalization {bool} -- Flag indicating whether to normalize votes by degree

    Returns:
        np.ndarray -- Votes for each candidate
    """

    votes = np.dot(candidates, np.dot(voters, voter_id))
    if node_degree_normalization:
        node_degree = np.sum(voter_id, axis=0)
        votes += node_degree
        norm = np.dot(candidates, np.sum(voters, axis=1)) + voters.shape[1]
        votes /= norm[:, None]
    return votes


def score_default(
    X, S, C, node_degree_normalization, means=True
):
    """Compute ROCs according to the default procedure

     Default procedure computes ranked cell similarity matrix and then uses neighbor voting

    Arguments:
         X {array} -- Array (sparse or dense) of geneset x cells
         S {vector} -- Study labels, length cells
         C {vector} -- Cell type labels, legnth cells
         node_degree_normalization {bool} -- Flag for whether to normalize votes by node degree

     Returns:
         pd.Series -- Series containing AUROCs for each cell type for the given gene set
    """
    nw = create_nw_spearman(X.T)
    nw = (nw + nw.T) / 2

    cell_labels = design_matrix(C)

    x1 = cell_labels.shape[1]
    x2 = cell_labels.shape[0]

    studies = np.unique(S)
    exp_cols = np.repeat(studies, x1)

    test_cell_labels = np.tile(cell_labels.values, studies.shape[0])
    for study in studies:  # Hide testing labels
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
        mask = (study != S).astype(float)[:, None] @ (study == exp_cols).astype(float)[
            :, None
        ].T
        mask = mask.astype(bool)
        filter_mat[mask] = np.nan
        predicts[mask] = np.nan

    predicts = bottleneck.nanrankdata(np.abs(predicts), axis=0)
    predicts[filter_mat == 0] = 0

    n_p = bottleneck.nansum(filter_mat, axis=0)
    n_n = bottleneck.nansum((filter_mat == 0).astype(float), axis=0)
    p = bottleneck.nansum(predicts, axis=0)
    rocNV = (p / n_p - (n_p + 1) / 2) / n_n

    # C array opposite of F in R
    rocNV = rocNV.reshape([studies.shape[0], x1]).T
    if means:
        return pd.Series(bottleneck.nanmean(rocNV, axis=1), index=cell_labels.columns)
    else:
        return pd.DataFrame(rocNV, index=cell_labels.columns, columns=studies)
