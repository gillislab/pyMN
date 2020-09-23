from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np 
from upsetplot import plot as UpSet
from .utils import *


def compute_nw_linkage(nw, method='average', **kwargs):
    nw2 = (nw + nw.T) / 2
    nw2.fillna(0, inplace=True)
    return hierarchy.linkage((1 - nw2.values)[np.triu_indices(nw2.shape[0],
                                                              1)],
                             method=method,
                             **kwargs)


def plotMetaNeighborUS(data,
                       threshold=None,
                       draw_rownames=False,
                       mn_key='MetaNeighborUS',
                       show=True,
                       figsize=(6, 6),
                       **kwargs):

    if type(data) is AnnData:
        assert mn_key in data.uns_keys(
        ), 'Must Run MetaNeighbor before plotting or pass results dataframe for data'
        df = data.uns[mn_key]
    else:
        df = data

    l = compute_nw_linkage(df)
    if threshold is not None:
        cm = sns.clustermap(df >= threshold,
                            row_linkage=l,
                            col_linkage=l,
                            figsize=figsize,
                            square=True,
                            **kwargs)
    else:
        cm = sns.clustermap(df,
                            row_linkage=l,
                            col_linkage=l,
                            figsize=figsize,
                            square=True,
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
    print(np.sum(auroc))
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index('study_ct')
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


def plotUpset(adata, study_col, ct_col, metaclusters, outlier_label = 'outliers', show=True):

    assert study_col in adata.obs_keys(), 'Study Col not in adata'
    assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index('study_ct')
    
    get_studies = lambda x: pheno.loc[x, study_col].values.tolist()
    studies = [get_studies(x) for x in metaclusters.values]
    membership  = dict(zip(metaclusters.index, studies))
    df = pd.DataFrame([{name: True
                    for name in names} for names in membership.values()],
                  index=membership.keys())
    df = df.fillna(False)
    df = df.loc[df.index != outlier_label]
    df = df.groupby(df.columns.tolist(), as_index=False).size()

    us = UpSet(df,
      sort_categories_by=None,
      sort_by='cardinality')
    us['intersections'].set(ylabel='# of Meta Clusters')
    if show:
        plt.show()
    else:
        return us
