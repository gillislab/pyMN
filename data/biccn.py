import os
from scanpy import read_10x_h5
from scipy import sparse
from anndata import AnnData, concat
import gc
import h5py

joint_url = "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/scell/10X/processed/analysis/RNASeq_integrated/"
# C = Chromium 10X, SS = SmartSeq
data_url = {
    'scCv2':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/scell/10X/processed/analysis/10X_cells_v2_AIBS/",
    'scCv3':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/scell/10X/processed/analysis/10X_cells_v3_AIBS/",
    'snCv2':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/sncell/10X/processed/analysis/10X_nuclei_v2_AIBS/",
    'snCv3Z':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/sncell/10X/processed/analysis/10X_nuclei_v3_AIBS/",
    'snCv3M':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/sncell/10X/processed/analysis/10X_nuclei_v3_Broad/",
    'scSS':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/scell/SMARTer/processed/analysis/SMARTer_cells_MOp/",
    'snSS':
    "http://data.nemoarchive.org/biccn/lab/zeng/transcriptome/sncell/SMARTer/processed/analysis/SMARTer_nuclei_MOp/"
}


@np.vectorize
def split_barcode(x):
    return x.split('L')[0]

@np.vectorize
def get_ds(x):
    ds_dict = {
        "10X_cells_v2_AIBS": "scCv2",
        "10X_cells_v3_AIBS": "scCv3",
        "10X_nuclei_v2_AIBS": "snCv2",
        "10X_nuclei_v3_AIBS": "snCv3Z",
        "10X_nuclei_v3_Broad": "snCv3M",
        "SmartSeq_cells_AIBS": "scSS",
        "SmartSeq_nuclei_AIBS": "snSS"
    }
    res = x.split('.')[0]
    return ds_dict[res]


@np.vectorize
def get_barcode(x):
    return x.split('.')[1]

def get_write_csv(dataset, url, key, write):
    df = pd.read_csv(f'{url}{key}', index_col=0)
    if write:
        df.to_csv(f'{dataset}/{key}')
    return df


def build_SS_adata(data_file):
    genes = data_file.index
    barcodes = data_file.columns
    expr = sparse.csr_matrix(data_file.values.T)
    adata = AnnData(expr,
                    var=pd.DataFrame(index=genes),
                    obs=pd.DataFrame(index=barcodes))
    return adata
def read_misformated_h5(dataset):
    f = h5py.File(f'{dataset}/umi_counts.h5', 'r')

    mat = f['matrix']
    idx = np.array(mat['indices'])
    idp = np.array(mat['indptr'])
    data = np.array(mat['data'])
    shape = np.array(mat['shape'])
    barcodes = np.array(mat['barcodes']).astype(str)
    features = np.array(mat['features']['name']).astype(str)

    expr = sparse.csc_matrix((data, idx, idp), shape=shape).T
    adata = AnnData(expr,
                    var=pd.DataFrame(index=features),
                    obs=pd.DataFrame(index=barcodes))
    f.close()
    return adata

@np.vectorize
def add_joint(x):
    return f'joint_{x}'

def get_joint_data(dataset):
    joint_ca = pd.read_csv(f'{joint_url}cluster.annotation.csv', index_col=0)
    joint_cm = pd.read_csv(f'{joint_url}cluster.membership.csv', index_col=0)
    joint_cm.columns = ['cluster_id']
    joint_data = pd.merge(joint_cm.reset_index(),
                          joint_ca,
                          how='inner',
                          on='cluster_id')
    joint_data.columns = add_joint(joint_data.columns.values)
    joint_data['study_id'] = get_ds(joint_data['joint_index'])
    joint_data['barcode'] = get_barcode(joint_data['joint_index'])

    joint_data.at[joint_data.study_id.isin(
        ['scCv2', 'scCv3', 'snCv2', 'snCv3Z']), 'barcode'] = split_barcode(
            joint_data.barcode[joint_data.study_id.isin(
                ['scCv2', 'scCv3', 'snCv2', 'snCv3Z'])].values)

    joint_data.set_index('barcode', inplace=True)
    return joint_data[joint_data.study_id == dataset]

def prepare_metadata(dataset, url, m, write):
    meta = get_write_csv(dataset, url, m, write)
    if dataset is 'snCv3M':
        meta.set_index('sample_name', inplace=True)
    else:
        meta.index.name = 'sample_name'

    ca = get_write_csv(dataset, url, 'cluster.annotation.csv', write)
    cm = get_write_csv(dataset, url, 'cluster.membership.csv', write)
    qc = get_write_csv(dataset, url, 'QC.csv', write)['x'].values
    cm.columns = ['cluster_id']
    ca_cm_merge = pd.merge(cm.reset_index(), ca, how='inner',
                           on='cluster_id').set_index('index')
    passed_qc = np.isfinite(pd.Series(
        1, index=qc)[ca_cm_merge.index.values]).values.astype(float)
    ca_cm_merge['passed_qc'] = passed_qc

    meta_merge = pd.concat([meta, ca_cm_merge], axis=1)
    meta_merge[meta_merge.passed_qc == True]
    if 'SS' not in dataset:
        #        meta_merge['sample_id'] = meta_merge.index.values.astype(str)
        meta_merge.index = split_barcode(meta_merge.index.values).astype(str)
    joint_data = get_joint_data(dataset)
    meta_merge = pd.concat([meta_merge, joint_data], axis=1)
    meta_merge = meta_merge.loc[:, ~meta_merge.columns.duplicated()]
    meta_merge['study_id'] = dataset
    return meta_merge

ds_genes = {}
for dataset in data_url.keys():
    print(dataset)
    url = data_url[dataset]
    try:
        os.mkdir(dataset)
    except:
        pass
    if 'SS' in dataset:
        m = 'sample_metadata.csv.gz'

        data_file = pd.read_csv(f'{url}exon.counts.csv.gz', index_col=0)

        adata = build_SS_adata(data_file)
        del data_file
        gc.collect()

    else:
        m = 'sample_metadata.csv'
        if dataset is 'snCv3M':
            adata = read_misformated_h5(dataset)
        else:
            adata = read_10x_h5(f'{dataset}/umi_counts.h5',
                                backup_url=f'{url}umi_counts.h5')
    meta_merge = prepare_metadata(dataset, url, m, False)
    meta_merge = meta_merge.loc[adata.obs_names.astype(str)]

    adata.obs = meta_merge
    adata.obs.columns = np.string_(adata.obs.columns)
    adata = adata[adata.obs[b'passed_qc'] == 1, :]
    if adata.var.shape[1] > 0:
        adata.var.columns = np.string_(adata.var.columns)
    ds_genes[dataset] = adata.var_names
    adata.write_h5ad(f'{dataset}/{dataset}.h5ad')

    del adata
    gc.collect()
shared_genes = np.array(
    list(set.intersection(*[set(x) for x in ds_genes.values()])))
common_data = [
    "sample_id", "cluster_id", "cluster_label", "subclass_label",
    "class_label", "cluster_color", "size", "passed_qc", "joint_cluster_id",
    "joint_cluster_label", "joint_cluster_color", "joint_subclass_id",
    "joint_subclass_label", "joint_subclass_color", "joint_class_id",
    "joint_class_label", "joint_class_color", "joint_cl", "joint_cluster_size",
    "joint_tree_order", "study_id"
]

adatas = []
for dataset in data_url.keys():
    print(dataset)
    adata = sc.read_h5ad(f'{dataset}/{dataset}.h5ad')
    adata = sc.read(f'{dataset}/{dataset}.h5ad')

    adata.obs.columns = adata.obs.columns.astype(str)

    adata.obs = adata.obs[common_data[1:]]
    adata.var_names_make_unique()
    adatas.append(adata)
    gc.collect()
joint_adata = concat(adatas, merge='same')
joint_adata.obs.columns = np.string_(joint_adata.obs.columns)

joint_adata.write_h5ad(
    'biccn_full.h5ad',
    compression='gzip',
    compression_opts=9)

gaba = joint_adata[joint_adata.obs.joint_class_label == 'GABAergic']

gaba.write_h5ad(
    'biccn_gaba.h5ad',
    compression='gzip',
    compression_opts=9)

hvgs = np.genfromtxt(
    'biccn_hvgs.csv',
    dtype=str)

adata_hvg = test_read[:, hvgs]

adata_hvg.write_h5ad(
    'biccn_hvg.h5ad',
    compression='gzip',
    compression_opts=9)