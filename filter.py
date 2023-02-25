import scanpy as sc
import h5py
import numpy as np

'''unscreen hypervariable genes'''
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
            if filter_min_counts:
                sc.pp.filter_genes(adata, min_counts=1)
                sc.pp.filter_cells(adata, min_counts=1)
            if size_factors or normalize_input or logtrans_input:
                adata.raw = adata.copy()
            else:
                adata.raw = adata
            if size_factors:
                 sc.pp.normalize_per_cell(adata)
                 adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
            else:
                adata.obs['size_factors'] = 1.0
            if logtrans_input:
               sc.pp.log1p(adata)
            if normalize_input:
               sc.pp.scale(adata)
            return adata

def load_data():
    print('loading data!')
    data_mat = h5py.File('your/csv/file/path.h5')
    X = np.array(data_mat['X'])
    data_mat.close() 
    adata = sc.AnnData(X)
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    return X,Y,adata.raw.X, adata.obs.size_factors


'''Screen hypervariable genes'''
##Read the data from the CSV file
#adata = sc.read_csv('your/csv/file/path.csv', first_column_names=True)
##Read the data from the H5 file
data_mat = h5py.File('your/csv/file/path.h5')
X = np.array(data_mat['X'])
Y = np.array(data_mat['Y'].value.astype('int'))
data_mat.close() 
adata = sc.AnnData(X)
# 数据预处理
'''
Each cell without at least 200 genes was filtered out using the sc.pp.filter_cells() function.
sc.pp.filter_genes() function was used to filter out each gene that was not present in at least three cells.
The counts of each cell were total-normalized using the sc.pp.normalize_total() function so that their total number was 1e4.
Log transformation was performed using the sc.pp.log1p() function.
The sc.pp.highly_variable_genes() function was used to identify highly variable genes, which included selecting the top 4000 genes with average expression values between 0.0125 and 3 and variance of gene expression values greater than 0.5.
Finally, highly variable genes were selected to construct new adata objects, and their shapes were printed out.
'''
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=4000,min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
# 数据保存到H5文件
sc.write('filename.h5ad', adata, compression='gzip')
