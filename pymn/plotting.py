from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np
sns.set(style='white', font_scale=1.25)


def compute_nw_linkage(nw, method='average', **kwargs):
    return hierarchy.linkage((1 - nw.values)[np.triu_indices(nw.shape[0], 1)],
                             method=method,
                             **kwargs)


def plotMetaNeighborUS(data,
                       threshold=None,
                       draw_rownames=False,
                       mn_key='MetaNeighborUS',
                       show=True,
                       figsize=(10, 6),
                       **kwargs):

    if type(data) is AnnData:
        assert mn_key in data.uns_keys(
        ), 'Must Run MetaNeighbor before plotting or pass results dataframe for data'
        df = data.uns[mn_key]
    else:
        df = data

    l = compute_nw_linkage(df, method='ward')
    if threshold is not None:
        cm = sns.clustermap(df >= threshold,
                            row_linkage=l,
                            col_linkage=l,
                            figsize=figsize,
                            square=True,
                            xticklabels=False,
                            yticklabels=False,
                            **kwargs)
    else:
        cm = sns.clustermap(df,
                            row_linkage=l,
                            col_linkage=l,
                            figsize=figsize,
                            square=True,
                            xticklabels=False,
                            yticklabels=False,
                            **kwargs)

    if show:
        plt.show()
    else:
        return cm


def plotMetaNeighbor(data,
                     ax=None,
                     mn_key='MetaNeighbor',
                     show=True,
                     figsize=(10, 6)):
    if type(data) is AnnData:
        assert mn_key in data.uns_keys(
        ), 'Must Run MetaNeighbor before plotting or pass results dataframe for data'
        df = data.uns[mn_key]
    else:
        df = data
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    df.index.name = 'Cell Type'
    df = pd.melt(df.reset_index(), id_vars='Cell Type', value_name='AUROC')
    fig, ax = plt.subplots

    #TODO: Implement colormap that comes from the cell type name
    sns.violinplot(data=df, x='Cell Type', y='AUROC', ax=ax)
    sns.swarmplot(data=df, x='Cell Type', y='AUROC', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
    sns.despine()

    if show:
        plt.show()
    else:
        return ax
