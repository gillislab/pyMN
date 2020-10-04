import numpy as np
import bottleneck
from scipy import sparse
import pandas as pd


def compute_var_genes(adata, return_vect=True):
    """Compute variable genes for an indiviudal dataset


    Arguments:
        adata {[type]} -- AnnData object containing a signle dataset

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    if sparse.issparse(adata.X):
        median = csc_median_axis_0(sparse.csc_matrix(adata.X))
    else:
        median = bottleneck.median(adata.X, axis=0)
    variance = np.var(adata.X.A, axis=0)
    bins = np.quantile(median, q=np.linspace(0, 1, 11), interpolation="midpoint")
    digits = np.digitize(median, bins, right=True)

    selected_genes = np.zeros_like(digits)
    for i in np.unique(digits):
        filt = digits == i
        var_tmp = variance[filt]
        bins_tmp = np.nanquantile(var_tmp, q=np.linspace(0, 1, 5))
        g = np.digitize(var_tmp, bins_tmp)
        selected_genes[filt] = (g >= 4).astype(float)

    if return_vect:
        return selected_genes.astype(bool)
    else:
        adata.var["highly_variable"] = selected_genes.astype(bool)


def variableGenes(adata, study_col, return_vect=False):
    """Comptue variable genes across data sets

    Identifies genes with high variance compared to their median expression
    (top quartile) within each experimentCertain function

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    assert study_col in adata.obs_keys(), "Study Col not in obs data"

    studies = np.unique(adata.obs[study_col])
    genes = adata.var_names
    var_genes_mat = pd.DataFrame(index=genes)

    for study in studies:
        slicer = adata.obs[study_col] == study
        genes_vec = compute_var_genes(adata[slicer])
        var_genes_mat.loc[:, study] = genes_vec.astype(bool)
    var_genes = np.all(var_genes_mat, axis=1)
    if return_vect:
        return var_genes
    else:
        adata.var["highly_variable"] = var_genes


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.
    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (
        _get_elem_at_rank(middle - 1, data, n_negative, n_zeros)
        + _get_elem_at_rank(middle, data, n_negative, n_zeros)
    ) / 2.0


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).
    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.
    """
    if not isinstance(X, sparse.csc_matrix):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):

        # Prevent modifying X in place
        data = np.copy(X.data[start:end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median
