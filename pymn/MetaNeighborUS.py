import numpy as np
import pandas as pd
import bottleneck

import gc

from .utils import *





def MetaNeighborUS(adata,
                   study_col,
                   ct_col,
                   var_genes='highly_variable',
                   symmetric_output=True,
                   node_degree_normalization=True,
                   fast_version=False,
                   one_vs_best=False,
                   trained_model=None,
                   save_uns=True,
                   mn_key='MetaNeighborUS'):

    assert study_col in adata.obs_keys(), 'Study Col not in adata'
    assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'

    assert np.unique(
        adata.obs[study_col].values).shape[0] > 1, 'Need more than 1 study'

    if var_genes is not 'highly_variable':
        var_genes = adata.var_names[np.in1d(adata.var_names, var_genes)]
    else:
        var_genes = adata.var_names[adata.var[var_genes]]
    assert var_genes.shape[0] > 2, 'Must have at least 2 genes'

    if trained_model is None:
        if fast_version:
            #Fast verion doesn't work with Categorical datatype
            assert adata.obs[
                study_col].dtype.name != 'category', 'Study Col is a category type, cast to either string or int'
            assert adata.obs[
                ct_col].dtype.name != 'category', 'Cell Type Col is a category type, cast to either string or int'

            cell_nv = metaNeighborUS_fast(adata[:, var_genes].X, adata.obs[study_col],
                                          adata.obs[ct_col],
                                          node_degree_normalization, one_vs_best)
        else:
            cell_nv = metaNeighborUS_default(adata[:, var_genes], study_col, ct_col,
                                             node_degree_normalization)
        if symmetric_output:
            cell_nv = (cell_nv + cell_nv.T) / 2
    else:
        cell_nv = MetaNeighborUS_from_trained(trained_model, adata[:, var_genes].X,
                                              adata.obs[study_col],
                                              adata.obs[ct_col],
                                              node_degree_normalization)

    cell_nv = cell_nv.astype(float)
    if save_uns:
        if one_vs_best:
            adata.uns[f'{mn_key}_1v1'] = cell_nv
        else:
            adata.uns[mn_key] = cell_nv
        adata.uns[f'{mn_key}_params'] = {
        'fast':fast_version,
        'node_degree_normalization':node_degree_normalization,
        'study_col':study_col,
        'ct_col':ct_col,
        'one_vs_best':one_vs_best,
        'symmetric_output':symmetric_output
        }
    else:
        return cell_nv


def metaNeighborUS_default(adata, study_col, ct_col,
                           node_degree_normalization):
    pheno, cell_labels, study_ct_uniq = create_cell_labels(
        adata, study_col, ct_col)

    rank_data = create_nw_spearman(adata.X.T)

    sum_in = rank_data @ cell_labels.values

    if node_degree_normalization:
        sum_all = np.sum(rank_data, axis=0)
        sum_in /= sum_all[:, None]

    cell_nv = compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col,
                                     ct_col)
    return cell_nv


def compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col, ct_col):
    cell_nv = pd.DataFrame(index=study_ct_uniq)
    for ct in study_ct_uniq:
        predicts_tmp = sum_in.copy()
        study, cellT = pheno[pheno.study_ct == ct].drop_duplicates()[[
            study_col, ct_col
        ]].values[0]  # Don't want to split string in case of charcter issues
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

        cell_nv[ct] = (p / n_p - (n_p + 1) / 2) / nn

        del predicts_tmp, filter_mat
        gc.collect()
    return cell_nv


def metaNeighborUS_fast(X, S, C, node_degree_normalization, one_vs_best):
    """
    X : cells x genes array or csr_matrix
    S : Study Label Pandas Series
    C : Cell Type label Pandas Series
    """

    #Makes it genes X cells
    X_norm = np.asfortranarray(normalize_cells(X).T)

    #Remove cells that have no variance
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
    LSC = pd.DataFrame({'study': S.values, 'cluster': C.values}, index=labels)

    result = predict_and_score(X_norm, LSC, cluster_centroids,
                               n_cells_per_cluster, labels_order,
                               node_degree_normalization, one_vs_best)
    result = result[result.index]
    return result


def predict_and_score(X_test, LSC, cluster_centroids, n_cells_per_cluster,
                      labels_order, node_degree_normalization, one_vs_best):

    if node_degree_normalization:
        centroid_study_label = LSC.drop_duplicates(
        ).loc[labels_order]['study'].values
        study_matrix = design_matrix(centroid_study_label)
        train_study_id = study_matrix.columns

        study_centroids = cluster_centroids.values @ study_matrix.values
        n_cells_per_study = n_cells_per_cluster @ study_matrix.values

    result = []
    S = LSC['study'].values
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
        aurocs = compute_aurocs(votes, positives=design_matrix(votes.index))
        if one_vs_best:
            aurocs = compute_1v1_aurocs(votes, aurocs)
        result.append(aurocs)
    return pd.concat(result)


def MetaNeighborUS_from_trained(trained_model, test_data, study_col, ct_col,
                                node_degree_normalization):
    dat = normalize_cells(test_data.X).T
    is_na = np.all(np.isfinite(dat), axis=0)
    dat = dat[:, ~is_na]
    cluster_centroids = train_model.iloc[1:]
    n_cells_per_cluster = train_model.iloc[1].values

    study_col = study_col[~is_na]
    ct_col = ct_col[~is_na]
    labels = join_labels(study_col, ct_col)
    LSC = pd.DataFrame({'study': study_col, 'cluster': ct_col}, index=labels)
    result = predict_and_score(dat, LSC,
                               cluster_centroids, n_cells_per_cluster,
                               np.unique(labels), node_degree_normalization)
    return result
