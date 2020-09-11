import numpy as np
from scipy import sparse
import pandas as pd
import bottleneck
import gc


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
    np.putmask(data, finite, ranks)
    del ranks, finite
    gc.collect()


def create_nw_spearman(data):
    if sparse.issparse(data):
        data = data.toarray()
    data = bottleneck.rankdata(data, axixs=0)
    nw = np.corrcoef(data, rowvar=False)
    rank(nw, nan_val=0)
    np.fill_diagonal(nw, 1)
    return nw

@np.vectorize
def join_labels(x, y):
    return f'{x}|{y}'


def design_matrix(vec):
    if type(vec) == pd.Series:
        vec = vec.values
    return pd.get_dummies(vec).set_index(vec).astype(float)


def normalize_cells(X, ranked=True):
    if sparse.issparse(X):
        res = X.toarray()
    else:
        res = X
    if ranked:
        res = bottleneck.rankdata(res, axis=1)

    avg = bottleneck.nanmean(res, axis=1)
    res -= avg[:, None]

    norm = np.sqrt(bottleneck.nansum(res**2, axis=1))[:, None]
    res /= norm
    return res


def compute_aurocs(votes, positives=None):
    """
    Votes: df with votes
    Positives: df with design matrix
    """
    res_col = votes.columns
    if positives is None:
        positives = design_matrix(votes.index)
    res_idx = positives.columns
    positives = positives.values

    n_pos = bottleneck.nansum(positives, axis=0)
    n_neg = positives.shape[0] - n_pos

    sum_pos_ranks = positives.T @ bottleneck.rankdata(votes.values, axis=0)
    result = sum_pos_ranks / n_pos[:, None]
    result -= (n_pos[:, None] + 1) / 2
    result /= n_neg[:, None]
    return pd.DataFrame(result, index=res_idx, columns=res_col)
