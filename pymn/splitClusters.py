import numpy as np 
from scipy.cluster import hierarchy
from scipy.spatial import distance
import pandas as pd

from .plotting import compute_nw_linkage
from anndata import AnnData

def splitClusters(data, k, mn_key='MetaNeighborUS', save_uns =True):
	if type(data) is AnnData:
		assert mn_key in data.uns_keys(), f'{mn_key} not found in uns, run MetaNeighborUS or pass mn_data'
		mn_scores = data.uns[mn_key]
	else:
		mn_scores = data
	is_na = ~np.all(np.isnan(mn_scores),axis=0)

	linkage = compute_nw_linkage(mn_scores.loc[is_na,is_na] )
	membership = hierarchy.cut_tree(linkage, n_clusters=k)
	membership_series = pd.Series(np.ravel(membership), index=mn_scores.index[is_na])
	res = [membership_series.index[membership_series == i].values for i in range(k)]
	
	if save_uns and type(data) is AnnData:
		data.uns[f'{mn_key}_split_{k}'] = res
	else:
		return res



def splitTrainClusters(data,k, mn_key='MetaNeighborUS',save_uns=True):
	if type(data) is AnnData:
		assert mn_key in data.uns_keys(), f'{mn_key} not found in uns, run MetaNeighborUS or pass mn_data'
		mn_scores = data.uns[mn_key]
	else:
		mn_scores = data
	row_is_na = np.all(np.isnan(mn_scores), axis=1)
	col_is_na = np.all(np.isnan(mn_scores), axis=0)

	mn_s = mn_scores.loc[~row_is_na, ~col_is_na]
	linkage = hierarchy.linkage(distance.pdist(mn_s.values).T,method='average')
	membership = hierarchy.cut_tree(linkage, n_clusters=k)
	membership_series = pd.Series(np.ravel(membership), index=mn_scores.index[~row_is_na])
	
	res = [membership_series.index[membership_series == i].values for i in range(k)]
	if save_uns and type(data) is AnnData:
		data.uns[f'{mn_key}_split_train_{k}'] = res
	else:
		return res

def splitTestClusters(data,k, mn_key='MetaNeighborUS',save_uns=True):
	if type(data) is AnnData:
		assert mn_key in data.uns_keys(), f'{mn_key} not found in uns, run MetaNeighborUS or pass mn_data'
		mn_scores = data.uns[mn_key]
	else:
		mn_scores = data
	
	res  = splitTrainClusters(mn_scores, k, save_uns=False)
	if save_uns and type(data) is AnnData:
		data.uns[f'{mn_key}_split_test_{k}'] = res
	else:
		return res