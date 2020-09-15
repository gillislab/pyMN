from scipy.cluster import hierarchy

def compute_nw_linkage(nw, method='average', **kwargs):
    return hierarchy.linkage((1 - nw.values)[np.triu_indices(nw.shape[0], 1)],
                             method=method,
                             **kwargs)


def plot_mn_output(mn_output, threshold=None, draw_rownames=False):
    l = compute_nw_linkage(mn_output, method='ward')
    if threshold is not None:
        sns.clustermap(mn_output >= threshold, row_linkage=l, col_linkage=l)
    else:
        sns.clustermap(mn_output, row_linkage=l, col_linkage=l)
