import numpy as np
import pandas as pd
import bottleneck
import warnings

import gc

#from .utils import *
from utils import *


def MetaNeighborUS(adata,
                   study_col,
                   ct_col,
                   var_genes="highly_variable",
                   symmetric_output=True,
                   node_degree_normalization=True,
                   fast_version=False,
                   one_vs_best=False,
                   trained_model=None,
                   save_uns=True,
                   compute_p=True,
                   mn_key="MetaNeighborUS"):
    """Runs Unsupervised version of MetaNeighbor

    When it is difficult to know how cell type labels compare across datasets this
    function helps users to make an educated guess about the overlaps without
    requiring in-depth knowledge of marker genes.


    The output is a cell type-by-cell type mean AUROC matrix, which is
    built by treating each pair of cell types as testing and training data for
    MetaNeighbor, then taking the average AUROC for each pair (NB scores will not
    be identical because each test cell type is scored out of its own dataset,
    and the differential heterogeneity of datasets will influence scores).
    If symmetric_output is set to FALSE, the training cell types are displayed
    as columns and the test cell types are displayed as rows.
    If trained_model was provided, the output will be a cell type-by-cell
    type AUROC matrix with training cell types as columns and test cell types
    as rows (no swapping of test and train, no averaging).

    Arguments:
       adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
       study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
       ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels

    Keyword Arguments:
        var_genes {str or vector} -- String for boolean column in adata.var that indicates highly variable
            genes or vector of highly variable genes (default: {'highly_variable'})
        symmetric_output {bool} --  Boolean indicating whether make square matrix output symmetric (default: {True})
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree (default: {True})
        fast_version {bool} -- boolean indicating whether to run fast approximate version (default: {False})
        one_vs_best {bool} --  boolean indicating whether to compute AUROCs as one vs best instead
            of one vs all (must also have fast_version = True or use pretrained model) (default: {False})
        trained_model {pd.DataFrame} -- A dataframe containing a trained model from pymn.trainModel
            or from the R vesion of MetaNeighbor::trainModel (default: {None})
        save_uns {bool} -- Boolean indicating whether to save in adata.uns[mn_key],
            when False returns cell type x cell type AUROCs dataframe (default: {True})
        mn_key {str} -- Key for saving in adata.uns[mn_key] (default: {'MetaNeighborUS'})
    """

    assert study_col in adata.obs_keys(), "Study Col not in adata"
    assert ct_col in adata.obs_keys(), "Cluster Col not in adata"

    if trained_model is not None:
        var_genes = adata.var_names[np.in1d(adata.var_names,
                                            trained_model.index)]
        trained_model = pd.concat([
            pd.DataFrame(trained_model.iloc[0]).T, trained_model.loc[var_genes]
        ])
    elif type(var_genes) is str:
        assert (
            var_genes in adata.var_keys()
        ), f"If passing a string ({var_genes}) for var names, it must be in adata.var_keys()"
        var_genes = adata.var_names[adata.var[var_genes]]
    else:
        var_genes = adata.var_names[np.in1d(adata.var_names, var_genes)]

    assert var_genes.shape[0] > 2, "Must have at least 2 genes"
    if var_genes.shape[0] < 5:
        warnings.warn("You should have at least 5 Variable Genes",
                      category=UserWarning)
    if one_vs_best:
        assert (fast_version or trained_model is not None
                ), "If you want to run in one_vs_best mode you must also \
         run in fast version mode or use a pretrained model"

    if trained_model is None:
        assert (np.unique(adata.obs[study_col].values).shape[0] >
                1), "Need more than 1 study"
        if fast_version:
            # Fast verion doesn't work with Categorical datatype
            assert (
                adata.obs[study_col].dtype.name != "category"
            ), "Study Col is a category type, cast to either string or int"
            assert (
                adata.obs[ct_col].dtype.name != "category"
            ), "Cell Type Col is a category type, cast to either string or int"

            cell_nv = metaNeighborUS_fast(adata[:, var_genes].X,
                                          adata.obs[study_col],
                                          adata.obs[ct_col],
                                          node_degree_normalization,
                                          one_vs_best, compute_p)
        else:
            cell_nv = metaNeighborUS_default(adata[:, var_genes], study_col,
                                             ct_col, node_degree_normalization,
                                             compute_p)
    else:
        cell_nv = MetaNeighborUS_from_trained(trained_model,
                                              adata[:, var_genes].X,
                                              adata.obs[study_col].values,
                                              adata.obs[ct_col].values,
                                              node_degree_normalization,
                                              one_vs_best, compute_p)
    if compute_p:
        cell_p = cell_nv[1]
        cell_nv = cell_nv[0]
        cell_p = cell_p.astype(float)

    cell_nv = cell_nv.astype(float)
    if symmetric_output and not one_vs_best:
        cell_nv = (cell_nv + cell_nv.T) / 2
    if save_uns:
        if one_vs_best:
            adata.uns[f"{mn_key}_1v1"] = cell_nv
        else:
            adata.uns[mn_key] = cell_nv
        adata.uns[f"{mn_key}_params"] = {
            "fast": fast_version,
            "node_degree_normalization": node_degree_normalization,
            "study_col": study_col,
            "ct_col": ct_col,
            "one_vs_best": one_vs_best,
            "symmetric_output": symmetric_output,
        }
        if compute_p:
            adata.uns[f'{mn_key}_pval'] = cell_p
    else:
        return cell_nv


def metaNeighborUS_default(adata, study_col, ct_col, node_degree_normalization,
                           compute_p):
    """Runs MetaNeighbor using Default Method



    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
        ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    pheno, cell_labels, study_ct_uniq = create_cell_labels(
        adata, study_col, ct_col)

    rank_data = create_nw_spearman(adata.X.T)

    sum_in = rank_data @ cell_labels.values

    if node_degree_normalization:
        sum_all = np.sum(rank_data, axis=0)
        sum_in /= sum_all[:, None]

    cell_nv = compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col,
                                     ct_col, compute_p)
    return cell_nv


def compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col, ct_col,
                           compute_p):
    """Helper function to compute AUROCs from votes matrix of cells


    Arguments:
        sum_in {np.ndarray} -- votes matrix, cells x cell types votes
        study_ct_uniq {vector} -- vector of study_id|cell_type labels
        pheno {pd.DataFrame} -- dataframe wtih study_ct, study_id and ct_col for all cells
        study_col {str} -- String name of study_col in pheno
        ct_col {str} -- Stirng name of cell type col in pheno

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    cell_nv = pd.DataFrame(index=study_ct_uniq)
    if compute_p:
        cell_p = pd.DataFrame(index=study_ct_uniq)
    for ct in study_ct_uniq:
        predicts_tmp = sum_in.copy()
        study, cellT = (pheno[pheno.study_ct == ct].drop_duplicates()[[
            study_col, ct_col
        ]].values[0])  # Don't want to split string in case of charcter issues
        slicer = pheno[study_col] == study
        pheno2 = pheno[slicer]
        predicts_tmp = predicts_tmp[slicer]
        predicts_tmp = bottleneck.nanrankdata(predicts_tmp, axis=0)

        filter_mat = np.zeros_like(predicts_tmp)
        filter_mat[pheno2.study_ct == ct] = 1

        predicts_tmp[filter_mat == 0] = 0

        n_p = bottleneck.nansum(filter_mat, axis=0)
        nn = filter_mat.shape[0] - n_p
        p = bottleneck.nansum(predicts_tmp, axis=0)
        roc = (p / n_p - (n_p + 1) / 2) / nn
        cell_nv[ct] = roc
        if compute_p:
            U = roc * n_p * nn
            Z = (np.abs(U - (n_p * nn / 2))) / np.sqrt(n_p * nn *
                                                       (n_p + nn + 1) / 12)
            P = stats.norm.sf(Z)
            cell_p[ct] = P
        del predicts_tmp, filter_mat
        gc.collect()
    if compute_p:
        return cell_nv, cell_p
    return cell_nv


def metaNeighborUS_fast(X, S, C, node_degree_normalization, one_vs_best,
                        compute_p):
    """Fast MetaNeighbor Approximation Helper function


    The fast version is vectorized according to the following equations
    (Note that the point of these equations is to *never* compute the cell-cell network
     by reordering the matrix operations):
     - INPUTS:
       + Q = test (Query) data (genes x cells)
       + R = train (Ref) data (genes x cells)
       + L = binary encoding of training cell types (Labels) (cells x cell types)
       + S = binary encoding of train Studies (cells x studies)
     - NOTATIONS:
       + X* = normalize_cols(X) ~ scale(colRanks(X)) denotes normalized data
              (Spearman correlation becomes a simple dot product on normalized data)
       + N = Spearman(Q,R) = t(Q*).R* is the cell-cell similarity network
       + CL = R*.L are the cell type centroids (in the normalized space)
       + CS = R*.S are the study centroids (in the normalized space)
       + 1.L = colSums(L) = number of cells per (train) cell type
       + 1.S = colSums(S) = number of cells per (train) study
     - WITHOUT node degree normalization
       + Votes = N.L = t(Q*).R*.L = t(Q*).CL
     - WITH node degree normalization
       + Network becomes N+1 to avoid negative values
       + Votes = (N+1).L = N.L + 1.L = t(Q*).CL + 1.L
       + Node degree = (N+1).S = t(Q*).CS + 1.S
       + Note: Node degree is computed independently for each train study.

    Arguments:
        X {array} -- cells x variableGenes matrix (dense or sparse)
        S {vector} -- vector of study_id labels
        C {vector} -- vector of cell type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Returns:
        pd.DataFrame --  ROCs for cell type x cell type labels
    """

    # Makes it genes X cells
    X_norm = np.asfortranarray(normalize_cells(X).T)

    # Remove cells that have no variance
    filter_cells = np.any(np.isnan(X_norm), axis=0)
    X_norm = X_norm[:, ~filter_cells]

    S = S[~filter_cells]
    C = C[~filter_cells]
    S_order = np.unique(S.values)
    C_order = np.unique(C.values)

    labels = join_labels(S.values, C.values)
    labels_matrix = design_matrix(labels)

    cluster_centroids = X_norm @ labels_matrix.values
    cluster_centroids = pd.DataFrame(cluster_centroids,
                                     columns=labels_matrix.columns)
    labels_order = labels_matrix.columns
    n_cells_per_cluster = np.sum(labels_matrix.values, axis=0)
    LSC = pd.DataFrame({"study": S.values, "cluster": C.values}, index=labels)

    result = predict_and_score(X_norm, LSC, cluster_centroids,
                               n_cells_per_cluster, labels_order,
                               node_degree_normalization, one_vs_best,
                               compute_p)
    if compute_p:
        aurocs = result[0]
        aurocs = aurocs[aurocs.index]

        p_vals = result[1]
        p_vals = p_vals[p_vals.index]
        return aurocs, p_vals

    result = result[result.index]
    return result


def predict_and_score(X_test,
                      LSC,
                      cluster_centroids,
                      n_cells_per_cluster,
                      labels_order,
                      node_degree_normalization,
                      one_vs_best,
                      compute_p,
                      pretrained=False):
    """[summary]

    [description]

    Arguments:
        X_test {np.ndarray} -- Normalized gene x cell expression
        LSC {pd.DataFrame} -- Dataframe with columns of study_col and ct_col and study_col|ct_col as index
        cluster_centroids {pd.DataFrame} -- Dataframe with genes x cell type centroids
        n_cells_per_cluster {vector} -- Vector in same order as cluster_centroids columns for number of cells per cluster
        labels_order {vector} -- Vector order for study_col|ct_col labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Keyword Arguments:
        pretrained {bool} -- Whether or not it is passing a pretrained model or not (default: {False})

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    if node_degree_normalization:
        if pretrained:
            get_study_id = np.vectorize(lambda x: x.split("|")[0])
            centroid_study_label = get_study_id(
                cluster_centroids.columns.values)
        else:
            centroid_study_label = (LSC.drop_duplicates().loc[labels_order,
                                                              "study"].values)
        study_matrix = design_matrix(centroid_study_label)
        train_study_id = study_matrix.columns
        study_centroids = cluster_centroids.values @ study_matrix.values
        n_cells_per_study = n_cells_per_cluster @ study_matrix.values

    result = []
    if compute_p:
        result_p = []
    S = LSC["study"].values
    for test_study in np.unique(S):
        is_test = S == test_study
        X_dataset = X_test[:, is_test]
        votes_idx = LSC.index[is_test]
        votes_cols = labels_order
        votes = np.asfortranarray(X_dataset.T @ cluster_centroids.values)
        if node_degree_normalization:
            votes += n_cells_per_cluster

            node_degree = np.asfortranarray(X_dataset.T @ study_centroids)
            node_degree += n_cells_per_study

            for train_study in np.unique(train_study_id):
                is_train = centroid_study_label == train_study
                norm = node_degree[:, train_study_id == train_study]
                votes[:, is_train] = votes[:, is_train] / norm
        votes = pd.DataFrame(votes, index=votes_idx, columns=votes_cols)

        aurocs = compute_aurocs(votes,
                                positives=design_matrix(votes.index),
                                compute_p=compute_p)
        if compute_p:
            result_p.append(aurocs[1])
            aurocs = aurocs[0]
        if one_vs_best:
            aurocs = compute_1v1_aurocs(votes, aurocs)
        result.append(aurocs)
    if compute_p:
        return pd.concat(result), pd.concat(result_p)
    return pd.concat(result)


def MetaNeighborUS_from_trained(trained_model, test_data, study_col, ct_col,
                                node_degree_normalization, one_vs_best,
                                compute_p):
    """MetaNeighbor from Pretrained model

    Runs MetaNeighbor using a pretrained model in the fast approximate version

    Arguments:
        trained_model {pd.DataFrame} -- Genes x Cell Type dataframe of model, with first row being number of cells per cell type in the model
        test_data {array} -- Genes x Cells expression data
        study_col {vector} -- vector of study_id labels
        ct_col {vector} -- vector of cell type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    dat = normalize_cells(test_data).T
    is_na = np.any(np.isnan(dat), axis=0)
    dat = dat[:, ~is_na]
    cluster_centroids = trained_model.iloc[1:]
    n_cells_per_cluster = trained_model.iloc[0].values
    study_col = study_col[~is_na]
    ct_col = ct_col[~is_na]
    labels = join_labels(study_col, ct_col, replace_bar=True)
    LSC = pd.DataFrame({"study": study_col, "cluster": ct_col}, index=labels)
    result = predict_and_score(
        dat,
        LSC,
        cluster_centroids,
        n_cells_per_cluster,
        cluster_centroids.columns,
        node_degree_normalization,
        one_vs_best,
        compute_p,
        pretrained=True,
    )
    return result
