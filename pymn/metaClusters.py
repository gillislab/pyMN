import numpy as np
import pandas as pd
import networkx as nx
from .utils import create_cell_labels
from anndata import AnnData


def extractMetaClusters(
    data: AnnData, mn_key="MetaNeighborUS_1v1", threshold=0, outlier_label="outliers"
):
    """Extract MetaClusters from MetaNeighborUS results

    Extract groups of reciprocal top hits form a 1vsBest AUROC matrix

    Note that meta-clusters are *not* cliques, but connected components, e.g.,
    if 1<->2 and 1<->3 are reciprocal top hits, {1, 2, 3} is a meta-cluster,
    independently from the relationship between 2 and 3.

    Arguments:
        data {AnnData} -- Anndata object containing MetaNeighborUS 1vBest results or 1vBest resutls dataframe

    Keyword Arguments:
        mn_key {str} -- Location in adata.uns with 1vBest results stored (default: {'MetaNeighborUS_1v1'})
        threshold {float} -- AUROC threshold for calling clusters as connected. Two clusters
            belonging to the same meta-cluster will have an AUROC >= than the threshold
            in both directions (since 1vBest AUROCs are not symmetric) (default: {0})
        outlier_label {str} -- Name to call outlier clusters (ones with no connections) (default: {'outliers'})

    Returns:
        None or pd.Series -- Saves result in adata.uns[f'{mn_key}_metaclusters'] or
            returns series with metacluster name as index and list for cluster membership as values
    """

    if type(data) is AnnData:
        assert (
            mn_key in data.uns_keys()
        ), f"{mn_key} not in unstrucuted data, run MetaNeighborUS in 1v1 fast mode or pass correct key"
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
            result[f"metacluster_{n_mc}"] = component
            n_mc += 1
        else:
            outliers.append(component[0])
    result[outlier_label] = np.array(outliers)

    if type(data) is AnnData:
        data.uns[f"{mn_key}_metaclusters"] = pd.Series(result)
    else:
        return pd.Series(result)


def make_graph(best_hits: pd.DataFrame, threshold=0):
    """Helper function to make a graph from AUROC matrix

    Makes graph from reciprocal top hits

    Arguments:
        best_hits {pd.DataFrame} -- AUROC Matrix of 1vBest resutls

    Keyword Arguments:
        threshold {float} -- AUROC threshold for calling clusters as connected. Two clusters
            belonging to the same meta-cluster will have an AUROC >= than the threshold
            in both directions (since 1vBest AUROCs are not symmetric) (default: {0}) (default: {0})

    Returns:
        nx.Graph -- Graph of reciprocal top hits
    """
    adj = np.zeros_like(best_hits)
    adj[best_hits > threshold] = 1
    adj *= adj.T  # Only reciprocal top hits remain True
    adj = pd.DataFrame(adj, index=best_hits.index, columns=best_hits.columns)
    return nx.from_pandas_adjacency(adj)


def score_meta_clusters(
    adata: AnnData,
    meta_clusters="MetaNeighborUS_1v1_metaclusters",
    best_hits="MetaNeighborUS_1v1",
    mn_key="MetaNeighborUS",
    study_col=None,
    ct_col=None,
    outlier_label="outliers",
    save_uns="MetaNeighborUS_metacluster_scores",
):
    """Provide Summary Statistics for metaclusters


    The function creates a DataFrame with columns:
     "meta_cluster" contains meta-cluster names,
     "clusters" lists the clusters belonging to each meta-cluster,
     "n_studies" is the number of studies spanned by the meta-cluster,
     "score" is the average similarity between meta-cluster members
     (average AUROC, NAs are treated as 0).

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together,
            also must have run MetaNeighborUS in 1vBest mode and extractMetaClusters with results stored

    Keyword Arguments:
        meta_clusters {str} -- String Identifying the location of output from extractMetaClusters  (default: {'MetaNeighborUS_1v1_metaclusters'})
        best_hits {str} -- String Identifying the location of output from MetaNeighborUS in 1vBest mode (default: {'MetaNeighborUS_1v1'})
        mn_key {str} -- String Identifying the location of output from MetaNeighborUS (default: {'MetaNeighborUS'})
        study_col {[type]} -- If none, study col identified from adata.uns[f'{mn_key}_params']['study_col'] else pass a vector (default: {None})
        ct_col {[type]} -- If none, cell type identified from adata.uns[f'{mn_key}_params']['ct_col']  (default: {None})
        outlier_label {str} -- String defining the outlier label from extractMetaClusters (default: {'outliers'})
        save_uns {str or bool} -- If string save results under adata.uns[save_uns], if False return results (default: {'MetaNeighborUS_metacluster_scores'})
    """

    if type(meta_clusters) is str:
        assert (
            meta_clusters in adata.uns_keys()
        ), f"{meta_clusters} not in unstrucuted data, run extractMetaClusters or pass output of that function"
        meta_clusters = adata.uns[meta_clusters]
    if type(best_hits) is str:
        assert (
            best_hits in adata.uns_keys()
        ), f"{best_hits} not in unstrucuted data, run MetaNeighborUS in 1v1 fast mode or pass the output of that function"
        best_hits = adata.uns[best_hits]
    if study_col is None:
        study_col = adata.uns[f"{mn_key}_params"]["study_col"]
    if ct_col is None:
        ct_col = adata.uns[f"{mn_key}_params"]["ct_col"]

    bh = best_hits.copy()
    bh.fillna(0, inplace=True)

    modules = meta_clusters[meta_clusters.index != outlier_label]

    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index("study_ct")
    mc_summary = {}
    for module in modules.index:
        mc_summary[module] = {
            "clusters": modules[module],
            "n_studies": pheno.loc[modules[module], study_col].unique().shape[0],
            "score": np.nanmean(best_hits.loc[modules[module], modules[module]].values),
        }
    mc_summary[outlier_label] = {
        "clusters": ";".join(meta_clusters[outlier_label]),
        "n_studies": 1,
        "score": np.nan,
    }
    if bool(save_uns):
        adata.uns[save_uns] = pd.DataFrame(mc_summary).T
    else:
        return pd.DataFrame(mc_summary).T
