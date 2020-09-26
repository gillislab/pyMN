import scanpy as sc 
import numpy as np
import pandas as pd

hemberg = sc.read_mtx(
    '/tyrone-data/bharris/metaneighbor_protocol_data/pancreas.mtx')
hemberg_coldata = pd.read_csv(
    '/tyrone-data/bharris/metaneighbor_protocol_data/pancreas_col.csv',
    index_col=0)
hemberg_genes = np.genfromtxt(
    '/tyrone-data/bharris/metaneighbor_protocol_data/pancreas_genes.csv',
    dtype=str)

hemberg = hemberg.T

hemberg.obs = hemberg_coldata

hemberg.var_names = hemberg_genes
hemberg.obs.columns = np.string_(hemberg.obs.columns)
hemberg.write_h5ad(
    '/tyrone-data/bharris/metaneighbor_protocol_data/hemberg.h5ad',
    compression='gzip',
    compression_opts=9)