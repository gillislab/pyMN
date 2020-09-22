import numpy as np 
from scipy.cluster import hierarchy
from scipy.spatial import distance
import pandas as pd

from .plotting import compute_nw_linkage


def splitClusters(mn_scores, k):
	is_na = ~np.all(np.isnan(mn_scores),axis=0)

	linkage = compute_nw_linkage(mn_scores.loc[is_na,is_na] )
	membership = hierarchy.cut_tree(linkage, n_clusters=k)
	membership_series = pd.Series(np.ravel(membership), index=mn_scores.index[is_na])
	return [membership_series.index[membership_series == i].values for i in range(k)]


def splitTrainClusters(mn_scores,k):
	row_is_na = np.all(np.isnan(mn_scores), axis=1)
	col_is_na = np.all(np.isnan(mn_scores), axis=0)

	mn_s = mn_scores.loc[~row_is_na, ~col_is_na]
	linkage = hierarchy.linkage(distance.pdist(mn_s.values).T,method='average')
	membership = hierarchy.cut_tree(linkage, n_clusters=k)
	membership_series = pd.Series(np.ravel(membership), index=mn_scores.index[is_na])
	return [membership_series.index[membership_series == i].values for i in range(k)]

def splitTestClusters(mn_scores, k):
	return splitTrainClusters(mn_scores.T, k)