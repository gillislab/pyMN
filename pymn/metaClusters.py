import numpy as np
import pandas as pd
import networkx as nx
from .utils import create_cell_labels


def extractMetaClusters(best_hits, threshold=0):
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
    result['outliers'] = np.array(outliers)
    return pd.Series(result)


def make_graph(best_hits, threshold=0):
    adj = np.zeros_like(best_hits)
    adj[best_hits > threshold] = 1
    adj = adj @ adj.T
    adj = pd.DataFrame(adj, index=best_hits.index, columns=best_hits.columns)
    return nx.from_pandas_adjacency(adj)


def score_meta_clusters(meta_clusters,
                        best_hits,
                        adata,
                        study_col,
                        ct_col,
                        outlier_label='outliers'):

    assert study_col in adata.obs_keys(), 'Study Col not in adata'
    assert ct_col in adata.obs_keys(), 'Cluster Col not in adata'

    bh = best_hits.copy()
    bh.fillna(0, inplace=True)

    modules = meta_clusters[meta_clusters.index != outlier_label]

    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index('study_ct')
    mc_summary = {}
    for module in modules.index:
        mc_summary[module] = {
            'meta_cluster':
            module,
            'clusters':
            modules[module],
            'n_studies':
            pheno.loc[modules[module], study_col].unique().shape[0],
            'score':
            np.mean(best_hits.loc[modules[module], modules[module]].values)
        }
    mc_summary[outlier_label] = {
        'meta_cluster': outlier_label,
        'clusters': ';'.join(meta_clusters[outlier_label]),
        'n_studies': 1,
        'score': np.nan
    }
    return pd.DataFrame(mc_summary)
