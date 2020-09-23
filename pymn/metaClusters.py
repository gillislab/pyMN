import numpy as np
import pandas as pd
import networkx as nx
from .utils import create_cell_labels
from anndata import AnnData

def extractMetaClusters(data, mn_key='MetaNeighborUS_1v1', threshold=0, outlier_label='outliers'):

    if type(data) is AnnData:
        assert mn_key in data.uns_keys(), f'{mn_key} not in unstrucuted data, run MetaNeighborUS in 1v1 fast mode or pass correct key'
        best_hits = data.uns[mn_key]
    else:
        best_hits = data

    comp = [
        np.array(list(l))
        for l in nx.connected_components(make_graph(best_hits, threshold))
    ]
    result = {}
    outliers = []
    n_mc = 1
    for component in comp:
        if component.shape[0] > 1:
            result[f'metacluster_{n_mc}'] = component
            n_mc += 1
        else:
            outliers.append(component[0])
    result[outlier_label] = np.array(outliers)
    
    if type(data) is AnnData:
        data.uns[f'{mn_key}_metaclusters'] = pd.Series(result)
    else:
        return pd.Series(result)


def make_graph(best_hits, threshold=0):
    adj = np.zeros_like(best_hits)
    adj[best_hits > threshold] = 1
    adj *=  adj.T
    adj = pd.DataFrame(adj, index=best_hits.index, columns=best_hits.columns)
    return nx.from_pandas_adjacency(adj)


def score_meta_clusters(adata,
                        meta_clusters='MetaNeighborUS_1v1_metaclusters',
                        best_hits='MetaNeighborUS_1v1',
                        mn_key = 'MetaNeighborUS',
                        study_col=None,
                        ct_col=None,
                        outlier_label='outliers',
                        save_uns='MetaNeighborUS_metacluster_scores'):

    
    if type(meta_clusters) is str:
        assert meta_clusters in adata.uns_keys(), f'{mn_key} not in unstrucuted data, run extractMetaClusters or pass output of that function'
        meta_clusters = adata.uns[meta_clusters]
    if type(best_hits) is str:
        assert best_hits in adata.uns_keys(), f'{mn_key} not in unstrucuted data, run MetaNeighborUS in 1v1 fast mode or pass the output of that function'
        best_hits = adata.uns[best_hits]
    if study_col is None:
        study_col = adata.uns[mn_key]['study_col']
    if ct_col is None:
        ct_col = adata.uns[mn_key]['ct_col']

    bh = best_hits.copy()
    bh.fillna(0, inplace=True)

    modules = meta_clusters[meta_clusters.index != outlier_label]

    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index('study_ct')
    mc_summary = {}
    for module in modules.index:
        mc_summary[module] = {
            'clusters':
            modules[module],
            'n_studies':
            pheno.loc[modules[module], study_col].unique().shape[0],
            'score':
            np.nanmean(best_hits.loc[modules[module], modules[module]].values)
        }
    mc_summary[outlier_label] = {
        'clusters': ';'.join(meta_clusters[outlier_label]),
        'n_studies': 1,
        'score': np.nan
    }
    if bool(save_uns):
        adata.uns[save_uns] = pd.DataFrame(mc_summary).T
    else:
        return pd.DataFrame(mc_summary).T
