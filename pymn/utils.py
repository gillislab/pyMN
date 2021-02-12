import numpy as np
from scipy import sparse, stats
import pandas as pd
import bottleneck
import gc
import warnings


def create_cell_labels(adata, study_col, ct_col):
    """Create Cell Labels and Study Cell Type Labels

    Utility function that takes anndata object and study column and cell type columns.
    Pheno is a dataframe with 3 columns, the study column, cell type column and a column with '|'.join([study,cell_type])
    Cell_labels is the design matrix of the study|cell type string

    Arguments:
        adata {Anndata} -- Anndata object containing single cell data
        study_col {str} -- column name in adta.obs for study id
        ct_col {str} -- column name in adata.obs for cell type

    Returns:
        (pd.DataFrame, np.ndarray, np.ndarray) -- Dataframe of study_col, ct_col and combined string,
                                                  design matrix for the combined label,
                                                  Set of combined labels in order for all other analysis to use
    """
    pheno = adata.obs[[study_col, ct_col]].copy()

    pheno.loc[:, "study_ct"] = join_labels(pheno[study_col].values,
                                           pheno[ct_col].values)
    study_ct_uniq = np.unique(pheno.study_ct)
    cell_labels = pd.get_dummies(pheno.study_ct)
    return pheno, cell_labels, study_ct_uniq


def rank(data, nan_val):
    """Rank normalize data

    Rank standardize inplace

    Does not return
    Arguments:
        data {np.array} -- Array of data
    """
    finite = np.isfinite(data)
    ranks = bottleneck.rankdata(data[finite]).astype(data.dtype)

    ranks -= 1
    top = np.max(ranks)
    ranks /= top
    data[...] = nan_val
    data[np.where(finite)] = ranks
    del ranks, finite
    gc.collect()


def create_nw_spearman(data):
    """Create Co-expreesion network

    Computes co-expression nnetwork using Spearman correaltion and then ranking the network

    Arguments:
        data {array} -- Cells x Genes array of floats, either dense or sparse

    Returns:
        np.ndarray -- co-expression nnetwork (2-D dense array)
    """
    if sparse.issparse(data):
        data = data.toarray()
    data = bottleneck.rankdata(data, axis=0)

    nw = np.corrcoef(data, rowvar=False)
    np.fill_diagonal(nw, 1)
    rank(nw, nan_val=0)
    np.fill_diagonal(nw, 1)
    return nw


@np.vectorize
def join_labels(x, y, replace_bar=False):
    """Join Study and cell type labels

    Vectorizied function for joining labels for study and cell type, '$STUDY|$CELLTYPE'
    It is best to not use a '|' in either the x or y vector, however it is most crucial to not have it in x
    Decorators:
        np.vectorize

    Arguments:
        x {vector} --  List that goes to the left of the |
        y {vector} -- List of items (same legnth as x) that goes to the right of |

    Keyword Arguments:
        replace_bar {bool} -- If true, will replace any occurance of the '|' character in the x vector (default: {False})

    Returns:
        nd.ndarray -- array of joined strings from x and y
    """
    if replace_bar:
        warnings.warn("Replacing any | with a . in study column values")
        a = x.replace("|", ".")
        return f"{a}|{y}"
    else:
        return f"{x}|{y}"


def design_matrix(vec):
    """Create design matrix from a vector

    Creates design matrix of observations x features of floats (0,1)

    If passing series, the index is ignored

    Arguments:
        vec {Vector} -- np.ndarray or pd.Series of observations

    Returns:
        pd.DataFrame -- Dataframe of design matrix. Observations x Labels
    """
    if type(vec) == pd.Series:
        vec = vec.values
    return pd.get_dummies(vec).set_index(vec).astype(float)


def normalize_cells(X, ranked=True):
    """

    Scale matrix sthat all cells (rows) sum to 1 and have l2-norm of 1

    Arguments:
        X {array} -- Cell x gene matrix (sparse or dense)

    Keyword Arguments:
        ranked {bool} -- Indicator whether to rank cells (default: {True})

    Returns:
        np.ndarray -- Cells x genes matrix of normalized cells
    """
    if sparse.issparse(X):
        res = X.toarray()
    else:
        res = X
    if ranked:
        res = bottleneck.rankdata(res, axis=1)

    avg = np.mean(res, axis=1)
    res -= avg[:, None]

    norm = np.sqrt(bottleneck.nansum(res**2, axis=1))[:, None]
    res /= norm
    return res


def compute_aurocs(votes, positives=None, compute_p=False):
    """Compute AUORCs based on neighbors voting and candidates identities


    Arguments:
        votes {pd.DataFrame} -- DataFrame with votes for cell types
        

    Keyword Arguments:
        positives {Vector} -- Vector of assignments for positive values. If left empty,
        cells are assumed to be the row names of the votes matrix (default: {None})
        compute_p {pd.DataFrame} -- Boolean for whether or not to compute the p value for the AUROC (default : {False})
    Returns:
        pd.DataFrame -- DataFrame of testing cell types x training cell types
    """
    res_col = votes.columns
    if positives is None:
        positives = design_matrix(votes.index)
    res_idx = positives.columns
    positives = positives.values

    n_pos = bottleneck.nansum(positives, axis=0)
    n_neg = positives.shape[0] - n_pos

    sum_pos_ranks = positives.T @ bottleneck.rankdata(votes.values, axis=0)
    roc = sum_pos_ranks / n_pos[:, None]
    roc -= (n_pos[:, None] + 1) / 2
    roc /= n_neg[:, None]

    if compute_p:
        n_pos = n_pos[:, None]
        n_neg = n_neg[:, None]

        U = roc * n_pos * n_neg
        Z = (np.abs(U -
                    (n_pos * n_neg / 2))) / np.sqrt(n_pos * n_neg *
                                                    (n_pos + n_neg + 1) / 12)
        p = stats.norm.sf(Z)
        p = pd.DataFrame(p, index=res_idx, columns=res_col)
        return pd.DataFrame(roc, index=res_idx, columns=res_col), p
    return pd.DataFrame(roc, index=res_idx, columns=res_col)


def compute_1v1_aurocs(votes, aurocs):
    """Compute 1v1 AUROCs based on one-vs-best setting

    Iterates through voters and finds their top candidates
    Arguments:
        votes {pd.DataFrame} -- One vs Rest Votes
        aurocs {pd.DataFrame} -- One vs Rest AUROCs

    Returns:
        pd.DataFrame -- Cell type x Cell type 1v1 AUROCs
    """
    res = pd.DataFrame(index=aurocs.index, columns=aurocs.columns)
    for col in aurocs.columns:
        if np.all(np.isnan(aurocs[col].values)):
            continue
        best, second, score = find_top_candidates(votes[col], aurocs[col])
        res.loc[best, col] = score
        res.loc[second, col] = 1 - score
    return res


def find_top_candidates(votes, aurocs):
    """Find best and second best candidate

    Helper function for compute_1v1_aurocs

    Arguments:
        votes {pd.Series} -- Votes vector 1vBest for cluster of interest
        aurocs {pd.Series} -- AUROC vector 1vsBest for cluster of interest

    Returns:
        str, str, float -- Name of best candidate, second best candidate and score for best candidate
    """
    candidates = aurocs.sort_values(ascending=False).head(5).index
    best = candidates[0]
    votes_best = votes[votes.index == best]
    score = 1
    second_best = candidates[1]
    for contender in candidates[1:]:
        votes_contender = votes[votes.index == contender]

        pos = design_matrix(
            np.repeat([1, 0], [votes_best.shape[0], votes_contender.shape[0]]))
        vt = pd.DataFrame(pd.concat([votes_best, votes_contender]))
        auroc = compute_aurocs(vt, positives=pos).values[1, 0]
        if auroc < 0.5:
            second_best = best
            best = contender
            score = 1 - auroc
            votes_best = votes_contender
        elif auroc < score:
            score = auroc
            second_best = contender

    return best, second_best, score
