import numpy as np

from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

from anndata import AnnData

from upsetplot import plot as UpSet

import networkx as nx

import scanpy as sc

from .utils import *

def compute_nw_linkage(nw, method='average', make_sym=True, **kwargs):
    if make_sym:
        nw2 = (nw + nw.T) / 2
        nw2.fillna(0, inplace=True)
    else:
        nw2 = nw
    return hierarchy.linkage((1 - nw2.values)[np.triu_indices(nw2.shape[0],
                                                              1)],
                             method=method,
                             **kwargs)


def plotMetaNeighborUS(data,
                       threshold=None,
                       mn_key='MetaNeighborUS',
                       show=True,
                       figsize=(6, 6),
                       fontsize=6,
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
    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(),
                                  fontsize=fontsize)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(),
                                  fontsize=fontsize)

    if show:
        plt.show()
    else:
        return cm


def order_rows_according_to_cols(M, alpha=1):
    M2 = M.values**alpha
    row_score = np.nansum(M2.T * np.arange(M2.shape[1])[:,None],
                                  axis=0) / np.nansum(M2, axis=1)
    return M.index[np.argsort(row_score)]


def plotMetaNeighborUS_pretrained(data,
                                  threshold=None,
                                  mn_key='MetaNeighborUS',
                                  show=True,
                                  figsize=(6, 6),
                                  fontsize=6,
                                  alpha_row=10,
                                  alpha_col=1,
                                  **kwargs):
    if type(data) is AnnData:
        assert mn_key in data.uns_keys(
        ), 'Must Run MetaNeighbor before plotting or pass results dataframe for data'
        df = data.uns[mn_key].copy()
    else:
        df = data.copy()
    col_l = hierarchy.linkage(df.fillna(0).values.T ** alpha_col,
                              method='average')
    row_order = order_rows_according_to_cols(df.fillna(0).iloc[:, hierarchy.leaves_list(col_l)], alpha=alpha_row)
    df = df.loc[row_order]

    if threshold is None:
        cm = sns.clustermap(df,
                            col_linkage=col_l,
                            row_cluster=False,
                            square=True,
                            figsize=figsize,
                            **kwargs)
    else:
        sns.clustermap(df >= threshold,
                       col_linkage=col_l,
                       row_cluster=False,
                       square=True,
                       figsize=figsize,
                       **kwargs)

    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(),
                                  fontsize=fontsize)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(),
                                  fontsize=fontsize)

    if show:
        plt.show()
    else:
        return cm


def plotMetaNeighbor(data,
                     ax=None,
                     mn_key='MetaNeighbor',
                     ct_col=None,
                     show=True,
                     color=None,
                     palette='Set2',
                     xtick_rotation=45):

    if type(data) is AnnData:
        assert mn_key in data.uns_keys(
        ), 'Must Run MetaNeighbor before plotting or pass results dataframe for data'
        df = data.uns[mn_key]
        ct_col=data.uns[f'{mn_key}_params']['ct_col']
    else:
        df = data

    df.index.name = 'Cell Type'
    df = pd.melt(df.reset_index(), id_vars='Cell Type', value_name='AUROC')
    
    fig, ax = plt.subplots()
    #TODO: Implement colormap that comes from the cell type name
    if color == 'Cell Type' and AnnData:
        if f'{ct_col}_colors_dict' not in data.uns_keys():
            cell_types = np.unique(data.obs[ct_col])
            pal = sns.color_palette(palette, cell_types.shape[0])
            color_pal = pd.Series(pal, index=cell_types)
            data.uns[f'{ct_col}_colors_dict'] = color_pal
            color=color_pal
            hue='Cell Type'
            
        else:
            color_pal = data.uns[f'{ct_col}_colors_dict']
            color=color_pal
            hue='Cell Type'     
    else:
        color='.7'
        hue=None
        color_pal=None

    sns.violinplot(data=df, 
                   x='Cell Type', 
                   y='AUROC', 
                   ax=ax,
                   inner='quartile',
                   color=color,
                   palette=color_pal)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=xtick_rotation)
    sns.despine()

    if show:
        plt.show()
    else:
        return ax


def plotUpset(adata,
              study_col=None,
              ct_col=None,
              mn_key='MetaNeighborUS',
              metaclusters='MetaNeighborUS_1v1_metaclusters',
              outlier_label='outliers',
              show=True):

    if study_col is None:
        study_col = adata.uns[f'{mn_key}_params']['study_col']
    else:
        assert study_col in adata.obs_keys(), 'Study Col not in adata'
    if ct_col is None:
        ct_col = adata.uns[f'{mn_key}_params']['ct_col']
    else:
        assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'

    if type(metaclusters) is str:
        assert metaclusters in adata.uns_keys(
        ), 'Run extractMetaClusters or pass Metacluster Series'
        metaclusters = adata.uns[metaclusters]
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index('study_ct')

    get_studies = lambda x: pheno.loc[x, study_col].values.tolist()
    studies = [get_studies(x) for x in metaclusters.values]
    membership = dict(zip(metaclusters.index, studies))
    df = pd.DataFrame([{name: True
                        for name in names} for names in membership.values()],
                      index=membership.keys())
    df = df.fillna(False)
    df = df[df.index != outlier_label]
    df = df.groupby(df.columns.tolist(), as_index=False).size()
    cols = df.columns[:-1].copy()
    for col in cols:
        df.set_index(df[col], append=True, inplace=True)
    df.index = df.index.droplevel(0)
    df= df['size']
    us = UpSet(df, sort_by='cardinality')
    if show:
        plt.show()
    else:
        return us


def makeClusterGraph(adata,
                     best_hits='MetaNeighborUS_1v1',
                     low_threshold=0,
                     high_threshold=1,
                     save_graph='MetaNeighborUS_metacluster_graph'):
    if type(best_hits) is str:
        assert best_hits in adata.uns_keys(
        ), 'Run MetaNeighorUS in 1v1 mode to compute Best Hits'
        best_hits = adata.uns[best_hits]
    filtered_hits = best_hits.copy()
    filtered_hits.fillna(0, inplace=True)
    filtered_hits.values[(best_hits.values > hight_threshold) |
                         (best_hits.values < low_threshold)] = 0
    np.fill_diagonal(filtered_hits.values, 0)
    G = nx.from_pandas_adjacency(filtered_hits.T, nx.DiGraph)
    if bool(save_graph):
        adata.uns[save_graph] = G
    else:
        return G


def extendClusterSet(coi,
                     adata=None,
                     G='MetaNeighborUS_metacluster_graph',
                     max_neighbor_distance=2):
    if type(G) is str:
        assert adata is not None, 'Must pass AnnData object if not passing Graph'
        G = adata.uns[G]
    if isinstance(coi, str):
        result = set([coi])
    else:
        result = set(coi)
    for _ in range(2):
        to_add = []
        for v in result:
            to_add.extend(G.neighbors(v))
        result.update(set(to_add))
    return list(result)


def plotClusterGraph(adata,
                     G='MetaNeighborUS_metacluster_graph',
                     best_hits='MetaNeighborUS_1v1',
                     mn_key='MetaNeighborUS',
                     node_list=None,
                     study_col=None,
                     ct_col=None,
                     node_scale=1,
                     figsize=(6, 6),
                     font_size=10,
                     legend_loc='best',
                     show=True):
    if type(G) is str:
        assert G in adata.uns_keys(), 'Run Make Cluster Graph or Pass Graph'
        G = adata.uns[G]
    if type(best_hits) is str:
        assert best_hits in adata.uns_keys(
        ), 'Run MetaNeighborUS in fast 1v1 mode to create best_hits or pass it'
        best_hits = adata.uns[best_hits]
    if study_col is None:
        study_col = adata.uns[f'{mn_key}_params']['study_col']
        ct_col = adata.uns[f'{mn_key}_params']['ct_col']

    if node_list is not None:
        G = G.subgraph(node_list)
    #Compute Edge Color
    r, c = list(zip(*list(G.edges())))
    es = best_hits.lookup(c, r)
    es[np.isnan(es)] = 0
    ec = pd.cut(es, [0, .5, 1], labels=['orange', 'black'])

    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno.set_index('study_ct', inplace=True)
    pheno2 = pheno.drop_duplicates()
    ct_labels = dict(zip(list(G.nodes()), pheno2.loc[list(G.nodes()), ct_col]))
    study_labels = pheno2.loc[list(G.nodes()), study_col].values

    node_sizes = pd.cut(pheno.reset_index()['study_ct'].value_counts(),
                        [0, 10, 100, np.inf],
                        labels=[150, 300, 450])[list(
                            G.nodes())].astype(int).values * node_scale

    if f'{study_col}_colors_dict' not in adata.uns_keys():
        studies = np.unique(adata.obs[study_col])
        pal = sns.color_palette('Set2', studies.shape[0])
        color_pal = pd.Series(pal, index=studies)
        adata.uns[f'{study_col}_colors_dict'] = color_pal
    else:
        color_pal = adata.uns[f'{study_col}_colors_dict']

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(
        G,
        prog='neato',
        args=f'-Goverlap=true -size={figsize[0]},{figsize[0]}')
    nx.draw_networkx_nodes(G,
                           pos=pos,
                           ax=ax,
                           node_color=color_pal[study_labels].values,
                           node_size=node_sizes)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ec)
    nx.draw_networkx_labels(G, pos=pos, labels=ct_labels, font_size=font_size)
    ax.axis('off')

    #Prepare legend
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup, xdescent, ydescent, width,
                           height, fontsize, trans):
            return [
                plt.Line2D([width / 2], [height / 2.],
                           ls="",
                           marker=tup[1],
                           color=tup[0],
                           transform=trans)
            ]

    ax.legend(list(zip(color_pal.values, ['o'] * color_pal.shape[0])),
              color_pal.index,
              handler_map={tuple: MarkerHandler()},
              frameon=False,
              loc=legend_loc)
    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax
def plotDotPlot(adata,
                gene_set,
                normalize_library_size=True,
                mn_key='MetaNeighbor',
                study_col=None,
                ct_col=None,
                alpha_row=10,
                average_expressing_only=True,
                figsize=(10,6),
                fontsize=10,
                show=True):
    if study_col is None:
        study_col = adata.uns[f'{mn_key}_params']['study_col']
    else:
        assert study_col in adata.obs_keys(),'Must pass study col in obs keys'
    if ct_col is None:
        ct_col = adata.uns[f'{mn_key}_params']['ct_col']
    else:
        assert ct_col in adata.obs_keys(),'Must pass ct col in obs keys'
    
    gs = gene_set.index[gene_set.astype(bool)]
    gs = np.intersect1d(gs, adata.var_names)


    if normalize_library_size:
        expr = adata[:,gs].to_df().T
        expr /= np.ravel(adata.X.sum(axis=1)) * 1e6
        
    else:
        expr = adata[:,gs].to_df().T

    
    label_matrix = design_matrix(
        join_labels(adata.obs[study_col].values,
                    adata.obs[ct_col].values))
    label_matrix/=label_matrix.sum()
    centroids = pd.DataFrame(expr.fillna(0).values.astype(float) @  label_matrix.values.astype(float),
                             index=expr.index,
                             columns=label_matrix.columns)
    average_nnz = pd.DataFrame(((expr.values>0).astype(float) @ label_matrix.values),
                             index=expr.index,
                             columns=label_matrix.columns)
    if average_expressing_only:
        centroids/=average_nnz
    centroids = centroids.T.astype(float).apply(stats.zscore).T
    centroids.index.name='Gene'
    average_nnz.index.name='Gene'
    
    centroids.reset_index(inplace=True)
    average_nnz.reset_index(inplace=True)
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno.set_index('study_ct', inplace=True)
    pheno2 = pheno.drop_duplicates()
    
    centroids = pd.melt(centroids,id_vars='Gene',value_name="average_expression",var_name='study_ct')
    centroids.loc[:,'Cell Type'] = pheno2.loc[centroids['study_ct'].values, ct_col].values
    centroids = centroids.groupby(['Gene','Cell Type']).mean().reset_index()
    
    average_nnz = pd.melt(average_nnz,id_vars='Gene',value_name="percent_expressing",var_name='study_ct')
    average_nnz.loc[:,'Cell Type'] = pheno2.loc[average_nnz['study_ct'].values, ct_col].values
    average_nnz = average_nnz.groupby(['Gene','Cell Type']).mean().reset_index()
    
    result = centroids.merge(average_nnz, how='inner',on=['Gene','Cell Type'])
    
    row_order = order_rows_according_to_cols(pd.pivot(result,
                                                  index='Gene',
                                                  columns='Cell Type',
                                                  values='average_expression'),
                                         alpha=10)
    result.loc[:, 'Gene'] = pd.Categorical(result.Gene, categories=row_order)

    result.sort_values('Gene', inplace=True)

    result.loc[:, 'Gene'] = result['Gene'].astype(str)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=result.fillna(result['average_expression'].min() ),
                         x='Cell Type',
                         y='Gene',
                         hue='average_expression',
                         size='percent_expressing',
                         palette='RdYlBu_r',ax=ax)
    plt.yticks(fontsize=fontsize)
    ax.legend(loc=(1, 0), frameon=False)
    if show:
        plt.show()
    else:
        return ax
    