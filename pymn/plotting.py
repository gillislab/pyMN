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


def compute_nw_linkage(nw, method="average", make_sym=True, **kwargs):
    """Compute the Network Linkage for a similarity matrix

    Converts a similarity matrix to a distance matrix and computes linkage

    Arguments:
    nw {pd.DataFrame} -- DataFrame containing the similarity matrix
    **kwargs {[type]} -- Passed to scipy.cluster.hierarchy.linkage

    Keyword Arguments:
    method {str} -- linkage method for clustering (default: {'average'})
    make_sym {bool} -- [description] (default: {True})

    Returns:
    np.ndarray -- Linkage matrix
    """
    if make_sym:
        nw2 = (nw + nw.T) / 2
        nw2.fillna(0, inplace=True)
    else:
        nw2 = nw
    return hierarchy.linkage(
        (1 - nw2.values)[np.triu_indices(nw2.shape[0], 1)], method=method, **kwargs
    )


def plotMetaNeighborUS(
    data,
    threshold=None,
    mn_key="MetaNeighborUS",
    show=True,
    figsize=(6, 6),
    fontsize=6,
    **kwargs,
):
    """Plot results from MetaNeighborUS function

    Plots a clustered heatmap of the AUROCs from MetaNeighborUS

    Arguments:
      data {AnnData} -- AnnData object containing scRNAseq data and MetaNeighborUS results
      **kwargs {[type]} -- Passed to sns.clustermap

    Keyword Arguments:
      threshold {None or float} -- If None, plot continuous values,
        if float between 0 and 1 will threshold data and plot binary results (default: {None})
      mn_key {str} -- Location of MetaNeighborUS or MetaNeighborUS_1vBest results for plotting (default: {'MetaNeighborUS'})
      show {bool} -- Flag for indicating whether to show results or return ClusterGrid object (default: {True})
      figsize {tuple} -- Parameters for controlling figure size in inches (default: {(6, 6): (float, float)})
      fontsize {[type]} -- Parameters for controlling fontsize for x and y labels (default: {6 (float)})
    """

    if type(data) is AnnData:
        assert (
            mn_key in data.uns_keys()
        ), "Must Run MetaNeighbor before plotting or pass results dataframe for data"
        df = data.uns[mn_key]
    else:
        df = data

    l = compute_nw_linkage(df)
    if threshold is not None:
        cm = sns.clustermap(
            df >= threshold,
            row_linkage=l,
            col_linkage=l,
            figsize=figsize,
            square=True,
            **kwargs,
        )
    else:
        cm = sns.clustermap(
            df, row_linkage=l, col_linkage=l, figsize=figsize, square=True, **kwargs
        )
    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), fontsize=fontsize)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize=fontsize)

    if show:
        plt.show()
    else:
        return cm


def order_rows_according_to_cols(M, alpha=1.0):
    """Helper function for clustering rows

    Cluster rows to make high values along the diagonal

    Arguments:
    M {pd.DataFrame} -- Dataframe with columns order how they will be plotted

    Keyword Arguments:
    alpha {number} -- Power to raise dataframe to, higher gives more weight to extreme values (default: {1})

    Returns:
    np.ndarray -- 1-D array with order for rows
    """
    M2 = M.values ** alpha
    row_score = np.nansum(M2.T * np.arange(M2.shape[1])[:, None], axis=0) / np.nansum(
        M2, axis=1
    )
    return M.index[np.argsort(row_score)]


def plotMetaNeighborUS_pretrained(
    data,
    threshold=None,
    mn_key="MetaNeighborUS",
    show=True,
    figsize=(6, 6),
    fontsize=6,
    alpha_row=10,
    alpha_col=1,
    **kwargs):
    """[summary]

     Plots rectangular AUROC heatmap, clustering train cell types (columns)
     by similarity, and ordering test cell types (rows) according to similarity
     to train cell types..

    Arguments:
      data {AnnData} -- AnnData object containing scRNAseq data and MetaNeighborUS results
        or pd.DataFrame containing results
      **kwargs {[type]} -- Passed to sns.clustermap

    Keyword Arguments:
      threshold {None or float} -- If None, plot continuous values,
        if float between 0 and 1 will threshold data and plot binary results (default: {None})
      mn_key {str} -- Location of MetaNeighborUS or MetaNeighborUS_1vBest results for plotting (default: {'MetaNeighborUS'})
      show {bool} -- Flag for indicating whether to show results or return ClusterGrid object (default: {True})
      figsize {tuple} -- Parameters for controlling figure size in inches (default: {(6, 6): (float, float)})
      fontsize {[type]} -- Parameters for controlling fontsize for x and y labels (default: {6 (float)})
      alpha_row {number} -- Parameter controling row ordering: a higher value of
        alpha_row gives more weight to extreme AUROC values (close to 1) (default: {10})
      alpha_col {number} -- Parameter controling col ordering: a higher value of
        alpha_col gives more weight to extreme AUROC values (close to 1). (default: {1})
    """
    if type(data) is AnnData:
        assert (
            mn_key in data.uns_keys()
        ), "Must Run MetaNeighbor before plotting or pass results dataframe for data"
        df = data.uns[mn_key].copy()
    else:
        df = data.copy()
    col_l = hierarchy.linkage(df.fillna(0).values.T ** alpha_col, method="average")
    row_order = order_rows_according_to_cols(
        df.fillna(0).iloc[:, hierarchy.leaves_list(col_l)], alpha=alpha_row
    )
    df = df.loc[row_order]

    if threshold is None:
        cm = sns.clustermap(
            df,
            col_linkage=col_l,
            row_cluster=False,
            square=True,
            figsize=figsize,
            **kwargs,
        )
    else:
        sns.clustermap(
            df >= threshold,
            col_linkage=col_l,
            row_cluster=False,
            square=True,
            figsize=figsize,
            **kwargs,
        )

    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), fontsize=fontsize)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize=fontsize)

    if show:
        plt.show()
    else:
        return cm


def plotMetaNeighbor(
    data,
    mn_key="MetaNeighbor",
    ct_col=None,
    show=True,
    color=None,
    palette="Set2",
    xtick_rotation=45,
):
    """Plot MetaNeighbor results in violin plots

    Showing how replicability of cell tpye depends on the gene sets

    Arguments:
        data {AnnData} -- AnnData object containing scRNAseq data and MetaNeighbor results
            or dataframe containing the results

    Keyword Arguments:
        mn_key {str} -- Location of MetaNeighbor results for plotting (default: {'MetaNeighbor'})
        ct_col {[type]} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['ct_col'] else passed as vector (default: {None})
        show {bool} -- Flag for showing plot or return ax (default: {True})
        color {[type]} -- If "Cell Type" then colors each violin plot by a color from either
            an estabilished colormap stored in adata.uns[f'{ct_col}_colors_dict'] or created using the palette (default: {None})
        palette {str} -- Name of palette for crateing a colormap for cell types(default: {'Set2'})
        xtick_rotation {number} -- Angle of rotation for x-axis lables (0-360) (default: {45})

    ##TODO: Add param to pass HVG scores to plot as red point in each cell type

    """

    if type(data) is AnnData:
        assert (
            mn_key in data.uns_keys()
        ), "Must Run MetaNeighbor before plotting or pass results dataframe for data"
        df = data.uns[mn_key]
        ct_col = data.uns[f"{mn_key}_params"]["ct_col"]
    else:
        df = data

    df.index.name = "Cell Type"
    df = pd.melt(df.reset_index(), id_vars="Cell Type", value_name="AUROC")

    fig, ax = plt.subplots()
    if color == "Cell Type" and AnnData:
        if f"{ct_col}_colors_dict" not in data.uns_keys():
            cell_types = np.unique(data.obs[ct_col])
            pal = sns.color_palette(palette, cell_types.shape[0])
            color_pal = pd.Series(pal, index=cell_types)
            data.uns[f"{ct_col}_colors_dict"] = color_pal
            color = color_pal
            hue = "Cell Type"

        else:
            color_pal = data.uns[f"{ct_col}_colors_dict"]
            color = color_pal
            hue = "Cell Type"
    else:
        color = ".7"
        hue = None
        color_pal = None

    sns.violinplot(
        data=df,
        x="Cell Type",
        y="AUROC",
        ax=ax,
        inner="quartile",
        color=color,
        palette=color_pal,
    )
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=xtick_rotation)
    sns.despine()

    if show:
        plt.show()
    else:
        return ax


def plotUpset(
    adata,
    study_col=None,
    ct_col=None,
    mn_key="MetaNeighborUS",
    metaclusters="MetaNeighborUS_1v1_metaclusters",
    outlier_label="outliers",
    show=True):
    """Plot UpSet plot for intersections between datasets and metaclusters

    Shows how replicability depends on hte input dataset

    Arguments:
        adata {AnnData} -- AnnData object containing the output  of MetaNeighborUS 1vBest, and extractMetaClusters

    Keyword Arguments:
        study_col {[type]} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['study_col'] else passed as vector (default: {None})
        ct_col {[type]} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['ct_col'] else passed as vector (default: {None})
        mn_key {str} -- Location of MetaNeighborUS results (default: {'MetaNeighborUS'})
        metaclusters {str} -- Location of extractMetaClusters results (default: {'MetaNeighborUS_1v1_metaclusters'})
        outlier_label {str} -- Name of outlier_label in metaclusters (extractMetaClusters results) (default: {'outliers'})
        show {bool} -- Flag for showing plot or return UpSet object (default: {True})
    """

    if study_col is None:
        study_col = adata.uns[f"{mn_key}_params"]["study_col"]
    else:
        assert study_col in adata.obs_keys(), "Study Col not in adata"
    if ct_col is None:
        ct_col = adata.uns[f"{mn_key}_params"]["ct_col"]
    else:
        assert ct_col in adata.obs_keys(), "Cluster Col not in adata"

    if type(metaclusters) is str:
        assert (
            metaclusters in adata.uns_keys()
        ), "Run extractMetaClusters or pass Metacluster Series"
        metaclusters = adata.uns[metaclusters]
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno = pheno.drop_duplicates().set_index("study_ct")

    get_studies = lambda x: pheno.loc[x, study_col].values.tolist()
    studies = [get_studies(x) for x in metaclusters.values]
    membership = dict(zip(metaclusters.index, studies))
    df = pd.DataFrame(
        [{name: True for name in names} for names in membership.values()],
        index=membership.keys(),
    )
    df = df.fillna(False)
    df = df[df.index != outlier_label]
    df = df.groupby(df.columns.tolist(), as_index=False).size()
    cols = df.columns[:-1].copy()
    for col in cols:
        df.set_index(df[col], append=True, inplace=True)
    df.index = df.index.droplevel(0)
    df = df["size"]
    us = UpSet(df, sort_by="cardinality")
    if show:
        plt.show()
    else:
        return us


def makeClusterGraph(
    adata,
    best_hits="MetaNeighborUS_1v1",
    low_threshold=0,
    high_threshold=1,
    save_graph="MetaNeighborUS_metacluster_graph",
):
    """Make a grpah of the clusters based on similarity

    Converts AUROC matrix into a graph
    This representation is a useful alternative for heatmaps for large datasets
    and sparse AUROC matrices (MetaNeighborUS with one_vs_best = TRUE)


    Arguments:
        adata {AnnData} -- AnnData object containing MetaNeighborUS output

    Keyword Arguments:
        best_hits {str} -- [description] (default: {'MetaNeighborUS_1v1'})
        low_threshold {number} -- AUROC threshold min for including as an edge (default: {0})
        high_threshold {number} -- AUROC threshold max for including as an edge (default: {1})
        save_graph {str} -- IF string save string in adata.uns[save_graph],
            if False return nx.Graph (default: {'MetaNeighborUS_metacluster_graph'})
    """
    if type(best_hits) is str:
        assert (
            best_hits in adata.uns_keys()
        ), "Run MetaNeighorUS in 1v1 mode to compute Best Hits"
        best_hits = adata.uns[best_hits]
    filtered_hits = best_hits.copy()
    filtered_hits.fillna(0, inplace=True)
    filtered_hits.values[
        (best_hits.values > high_threshold) | (best_hits.values < low_threshold)
    ] = 0
    np.fill_diagonal(filtered_hits.values, 0)
    G = nx.from_pandas_adjacency(filtered_hits.T, nx.DiGraph)
    if bool(save_graph):
        adata.uns[save_graph] = G
    else:
        return G


def extendClusterSet(
    coi,
    adata=None,
    G="MetaNeighborUS_metacluster_graph",
    max_neighbor_distance=2):
    """Extend cluster set to nearest neighbor on cluster graph

    Note that the graph is directed, i.e. the neighbors are retrieved
    by following the arrows that start from the initial clusters

    Arguments:
        coi: {str or List[str]} -- Name of cluster or list of names of clusters to start search from

    Keyword Arguments:
        adata {AnnData} -- AnnData object containing nx.DiGraph to search from (default: {None})
        G {str or nx.DiGraph} -- Name of key in adata.uns to find graph or nx.DiGraph object
              (default: {'MetaNeighborUS_metacluster_graph')
        max_neighbor_distance {number} -- Path length from original coi(s) to search (default: {2})

    Returns:
        List : List of clusters in teh subgraph within a distance of max_neighbor_distance from the coi(s)
    """
    if type(G) is str:
        assert adata is not None, "Must pass AnnData object if not passing Graph"
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


def plotClusterGraph(
    adata,
    G="MetaNeighborUS_metacluster_graph",
    best_hits="MetaNeighborUS_1v1",
    mn_key="MetaNeighborUS",
    node_list=None,
    study_col=None,
    ct_col=None,
    node_scale=1,
    figsize=(6, 6),
    font_size=10,
    legend_loc="best",
    show=True,
):
    """Plot cluster graph generated with makeClusterGraph.

    In this visualization, edges are colored in black when AUROC > 0.5 and
    orange when AUROC < 0.5, edge width scales linearly with AUROC.
    Edges are oriented from training cluster towards
    test cluster. A black bidirectional edge indicates that two clusters are
    reciprocal top matches.
    Node radius reflects cluster size (small: up to 10 cells,
    medium: up to 100 cells, large: all other clusters).

    Arguments:
     adata {AnnData} -- AnnData object containing data and metadata

    Keyword Arguments:
     G {str or nx.DiGraph} -- Key in adata.uns for Graph or nx.DiGraph to plot (default: {'MetaNeighborUS_metacluster_graph'})
     best_hits {str or pd.DataFrame} -- Key in adata.uns From MetaNeighborUS 1vBest results or pd.DataFrame resutls (default: {'MetaNeighborUS_1v1'})
     mn_key {str} -- Key in adata.uns for MetaNeighborUS results (for getting params) (default: {'MetaNeighborUS'})
     node_list {List[str]} -- List of node to subset graph to if not plotting all clusters (default: {None})
     study_col {vector} -- If none, inferrefed from adata.uns[f'{mn_key}_params']['study_col'],
        else vector of study ids (default: {None})
     ct_col {vector} -- If none, inferrefed from adata.uns[f'{mn_key}_params']['ct_col'],
        else vector of cell type labels (default: {None})
     node_scale {number} -- Factor to scale node size up or down by (must be >0) (default: {1})
     figsize {tuple} -- Tuple to change figure size (default: {(6, 6): (float, float)})
     font_size {number} -- Font size for node labels (default: {10})
     legend_loc {str} -- Legend location either tuple or string based on matplotlib location names (default: {'best'})
     show {bool} -- If True shows plot, else returns ax (default: {True})
    """
    if type(G) is str:
        assert G in adata.uns_keys(), "Run Make Cluster Graph or Pass Graph"
        G = adata.uns[G]
    if type(best_hits) is str:
        assert (
            best_hits in adata.uns_keys()
        ), "Run MetaNeighborUS in fast 1v1 mode to create best_hits or pass it"
        best_hits = adata.uns[best_hits]
    if study_col is None:
        study_col = adata.uns[f"{mn_key}_params"]["study_col"]
        ct_col = adata.uns[f"{mn_key}_params"]["ct_col"]

    if node_list is not None:
        G = G.subgraph(node_list)
    # Compute Edge Color
    r, c = list(zip(*list(G.edges())))
    es = best_hits.lookup(c, r)
    es[np.isnan(es)] = 0
    ec = pd.cut(es, [0, 0.5, 1], labels=["orange", "black"])

    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno.set_index("study_ct", inplace=True)
    pheno2 = pheno.drop_duplicates()
    ct_labels = dict(zip(list(G.nodes()), pheno2.loc[list(G.nodes()), ct_col]))
    study_labels = pheno2.loc[list(G.nodes()), study_col].values

    node_sizes = (
        pd.cut(
            pheno.reset_index()["study_ct"].value_counts(),
            [0, 10, 100, np.inf],
            labels=[150, 300, 450],
        )[list(G.nodes())]
        .astype(int)
        .values
        * node_scale
    )

    if f"{study_col}_colors_dict" not in adata.uns_keys():
        studies = np.unique(adata.obs[study_col])
        pal = sns.color_palette("Set2", studies.shape[0])
        color_pal = pd.Series(pal, index=studies)
        adata.uns[f"{study_col}_colors_dict"] = color_pal
    else:
        color_pal = adata.uns[f"{study_col}_colors_dict"]

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(
        G, prog="neato", args=f"-Goverlap=true -size={figsize[0]},{figsize[0]}"
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=color_pal[study_labels].values,
        node_size=node_sizes,
    )
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ec)
    nx.draw_networkx_labels(G, pos=pos, labels=ct_labels, font_size=font_size)
    ax.axis("off")

    # Prepare legend
    class MarkerHandler(HandlerBase):
        def create_artists(
            self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
        ):
            return [
                plt.Line2D(
                    [width / 2],
                    [height / 2.0],
                    ls="",
                    marker=tup[1],
                    color=tup[0],
                    transform=trans,
                )
            ]

    ax.legend(
        list(zip(color_pal.values, ["o"] * color_pal.shape[0])),
        color_pal.index,
        handler_map={tuple: MarkerHandler()},
        frameon=False,
        loc=legend_loc,
    )
    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax


def plotDotPlot(
    adata,
    gene_set,
    normalize_library_size=True,
    mn_key="MetaNeighbor",
    study_col=None,
    ct_col=None,
    alpha_row=10,
    average_expressing_only=True,
    figsize=(10, 6),
    fontsize=10,
    show=True):
    """Plot dot plot showing expression of a gene set across cell types.

    The size of each dot reflects the number of cell that express a gene,
    the color reflects the average expression.
    Expression of genes is first average and scaled in each dataset
    independently. The final value is obtained by averaging across datasets.

    Arguments:
        adata {AnnData} -- AnnData object containing expression and metadata
        gene_set {pd.Series} -- Boolean pd.Series with genes as index and Genes that are in the geneset as True

    Keyword Arguments:
        normalize_library_size {bool} -- Wheter to normalize cells by library size (default: {True})
        mn_key {str} -- Locatoin of MetaNeighborUS results to get params data (default: {'MetaNeighbor'})
        study_col {vector} -- If none, inferrefed from adata.uns[f'{mn_key}_params']['study_col'],
            else vector of study ids (default: {None})
        ct_col {vector} -- If none, inferrefed from adata.uns[f'{mn_key}_params']['ct_col'],
            else vector of cell type labels (default: {None})
        alpha_row {number} -- Parameter controling row ordering: a higher value of
            alpha_row gives more weight to extreme AUROC values (close to 1) (default: {10})
        average_expressing_only {bool} -- Whether average expression should be computed based
            only on expressing cells (Seurat default) or taking into account zeros (default: {True})
        figsize {tuple} -- Tuple that sets figure size in inches (default: {(10,6)})
        fontsize {number} -- Fontsize of gene names in y axis ticks (default: {10})
        show {bool} -- If True shows plot, else returns ax (default: {True})
    """
    if study_col is None:
        study_col = adata.uns[f"{mn_key}_params"]["study_col"]
    else:
        assert study_col in adata.obs_keys(), "Must pass study col in obs keys"
    if ct_col is None:
        ct_col = adata.uns[f"{mn_key}_params"]["ct_col"]
    else:
        assert ct_col in adata.obs_keys(), "Must pass ct col in obs keys"

    gs = gene_set.index[gene_set.astype(bool)]
    gs = np.intersect1d(gs, adata.var_names)

    if normalize_library_size:
        expr = adata[:, gs].to_df().T
        expr /= np.ravel(adata.X.sum(axis=1)) * 1e6

    else:
        expr = adata[:, gs].to_df().T

    label_matrix = design_matrix(
        join_labels(adata.obs[study_col].values, adata.obs[ct_col].values)
    )
    label_matrix /= label_matrix.sum()
    centroids = pd.DataFrame(
        expr.fillna(0).values.astype(float) @ label_matrix.values.astype(float),
        index=expr.index,
        columns=label_matrix.columns,
    )
    average_nnz = pd.DataFrame(
        ((expr.values > 0).astype(float) @ label_matrix.values),
        index=expr.index,
        columns=label_matrix.columns,
    )
    if average_expressing_only:
        centroids /= average_nnz
    centroids = centroids.T.astype(float).apply(stats.zscore).T
    centroids.index.name = "Gene"
    average_nnz.index.name = "Gene"

    centroids.reset_index(inplace=True)
    average_nnz.reset_index(inplace=True)
    pheno, _, _ = create_cell_labels(adata, study_col, ct_col)
    pheno.set_index("study_ct", inplace=True)
    pheno2 = pheno.drop_duplicates()

    centroids = pd.melt(
        centroids, id_vars="Gene", value_name="average_expression", var_name="study_ct"
    )
    centroids.loc[:, "Cell Type"] = pheno2.loc[
        centroids["study_ct"].values, ct_col
    ].values
    centroids = centroids.groupby(["Gene", "Cell Type"]).mean().reset_index()

    average_nnz = pd.melt(
        average_nnz,
        id_vars="Gene",
        value_name="percent_expressing",
        var_name="study_ct",
    )
    average_nnz.loc[:, "Cell Type"] = pheno2.loc[
        average_nnz["study_ct"].values, ct_col
    ].values
    average_nnz = average_nnz.groupby(["Gene", "Cell Type"]).mean().reset_index()

    result = centroids.merge(average_nnz, how="inner", on=["Gene", "Cell Type"])

    row_order = order_rows_according_to_cols(
        pd.pivot(
            result, index="Gene", columns="Cell Type", values="average_expression"
        ),
        alpha=10,
    )
    result.loc[:, "Gene"] = pd.Categorical(result.Gene, categories=row_order)

    result.sort_values("Gene", inplace=True)

    result.loc[:, "Gene"] = result["Gene"].astype(str)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=result.fillna(result["average_expression"].min()),
        x="Cell Type",
        y="Gene",
        hue="average_expression",
        size="percent_expressing",
        palette="RdYlBu_r",
        ax=ax,
    )
    plt.yticks(fontsize=fontsize)
    ax.legend(loc=(1, 0), frameon=False)
    if show:
        plt.show()
    else:
        return ax
