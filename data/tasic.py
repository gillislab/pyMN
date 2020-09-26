import scanpy as sc 
import numpy as np
import pandas as pd

tasic = sc.read_mtx(
    'tasic_counts.mtx')
tasic_coldata = pd.read_csv(
    'tasic_col.csv',
    index_col=0)
tasic_genes = np.genfromtxt(
    'tasic_genes.csv',
    dtype=str)

tasic = tasic.T

tasic.obs = tasic_coldata

tasic.var_names = tasic_genes
tasic.obs.columns = np.string_(tasic.obs.columns)
tasic.write_h5ad(
    'tasic.h5ad',
    compression='gzip',
    compression_opts=9)