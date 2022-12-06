import numpy as np
import pandas as pd

def topHits(
    adata,
    cell_nv=None,
    mn_key="MetaNeighborUS",
    save_uns=True,
    threshold=0.95,
):
    """Find reciprocal top hits, stratifying results by study.

    Identifies reciprocal top hits and high scoring cell type pairs

    This function looks for reciprocal top hits in each target study
    separately, allowing for as many reciprocal top hits as target studies.

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together

    Keyword Arguments:
        study_col {str or vector} -- If None, inferrefed from adata.uns[f'{mn_key}_params']['study_col']
            else passed a vector (default: {None})
        mn_key {str} -- Location of MetaNeighborUS results (default: {'MetaNeighborUS'})
        save_uns {bool} --  Whether or not to save data in adata.uns[f'{mn_key}_topHits'] or return pd.DataFrame results (default: {True})
        threshold {number} -- AUROC threshold for calling a hit (default: {.95})
    """

    if cell_nv is not None:
        cnv = cell_nv.copy()
    if cell_nv is None:
        assert (
            mn_key in adata.uns_keys()
        ), "MetaNeighborUS resutls not stored in adata or passed as cell_nv"
        cnv = adata.uns[mn_key].copy()

    assert np.all(
        cnv.index == cnv.columns
    ), "cell_nv is does not have the same order in both the rows and columns"

    hit_matrix_melted = cnv.melt(ignore_index=False, value_name = "auroc", var_name = "target_cell_type").reset_index()
    hit_matrix_melted = hit_matrix_melted.rename(columns={'index':'ref_cell_type'})
    
    #could be sped up by doing this while it's a matrix
    hit_matrix_melted["ref_study"] = hit_matrix_melted["ref_cell_type"].str.replace("[|].*","",regex=True)
    hit_matrix_melted["target_study"] = hit_matrix_melted["target_cell_type"].str.replace("[|].*","",regex=True)
    
    #filter out same study hits
    hit_matrix_melted = hit_matrix_melted[hit_matrix_melted['target_study'] != hit_matrix_melted['ref_study']]
    #get best hit for each reference cell type and target study combination
    max_per_celltype_and_study = hit_matrix_melted.groupby(by=["ref_cell_type", "target_study"])["auroc"].max()
    max_per_celltype_and_study = pd.DataFrame(max_per_celltype_and_study).reset_index()
    max_per_celltype_and_study = pd.merge(hit_matrix_melted, max_per_celltype_and_study)
    reciprocal = max_per_celltype_and_study.drop(columns = "auroc")
    #flip the max scores to find reciprocal ones
    reverse_hits = reciprocal.rename(columns = {"ref_cell_type":"target_cell_type", 
                                                "target_cell_type":"reciprocal_cell_type"})
    
    reciprocal = pd.merge(reciprocal, reverse_hits, on = "target_cell_type")
    
    reciprocal["is_reciprocal"] = (reciprocal["ref_cell_type"] == reciprocal["reciprocal_cell_type"])
    
    #slim it down for joining
    reciprocal = reciprocal[["ref_cell_type", "target_cell_type", "is_reciprocal"]]
    
    max_per_celltype_and_study = pd.merge(max_per_celltype_and_study, reciprocal, how = "inner")
    
    #remove duplicates
    max_per_celltype_and_study = max_per_celltype_and_study.assign(pair_id = max_per_celltype_and_study[["ref_cell_type", "target_cell_type"]].values.tolist())
    #ensure same order
    max_per_celltype_and_study.pair_id = max_per_celltype_and_study.pair_id.apply(lambda x: sorted(x))
    #convert to string
    max_per_celltype_and_study.pair_id = max_per_celltype_and_study.pair_id.apply(lambda x: ','.join(map(str, x)))

    #threshold before removing duplicates - same position as in the R code
    max_per_celltype_and_study = max_per_celltype_and_study.loc[max_per_celltype_and_study.auroc >= threshold]

    #summarize/remove duplicates
    max_per_celltype_and_study = max_per_celltype_and_study.groupby("pair_id").agg({'ref_cell_type' :'first', 'target_cell_type':'first', 'auroc': 'mean', 'is_reciprocal' : 'first'})
    
    max_per_celltype_and_study = max_per_celltype_and_study.reset_index(drop=True)
    
    max_per_celltype_and_study['is_reciprocal'] = np.where(max_per_celltype_and_study['is_reciprocal'], "Reciprocal_top_hit", "Above_" + str(threshold))
    
    max_per_celltype_and_study = max_per_celltype_and_study.rename(columns = {"ref_cell_type":"Study_ID|Celltype_1", 
                                                 "target_cell_type":"Study_ID|Celltype_2",
                                                 "auroc": "Mean_AUROC", "is_reciprocal":"Match_type"})
    max_per_celltype_and_study.sort_values("Mean_AUROC", ascending=False, inplace=True)
    #filter based on threshold (after averaging)
    max_per_celltype_and_study.Mean_AUROC = np.round(max_per_celltype_and_study.Mean_AUROC, 2)
    if save_uns:
        adata.uns[f"{mn_key}_topHits"] = max_per_celltype_and_study
    else:
        return max_per_celltype_and_study
