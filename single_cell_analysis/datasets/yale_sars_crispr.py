import anndata
from scanpy.readwrite import read
from scanpy import AnnData
from scanpy import settings
import scanpy as sc
import pandas as pd
import numpy as np

from pathlib import Path
import glob

class YaleSarsCrispr():
    def __init__(self, data_path, preprocess=False, ignore_cache=False, write_cache=True, cache_dir=None):
        self._data_path = Path(data_path)
        self._data_path_sars = Path(data_path) / 'data' / 'SARS2_MON_crispr'
        self._data_path_mock = Path(data_path) / 'data' / 'MOCK_MON_crispr'
        
        self._preprocess = preprocess
        self._ignore_cache = ignore_cache

        cache_dir = Path(cache_dir) if cache_dir else self._data_path
        self._cache_path = 'data_processed.h5ad' if preprocess else 'data_raw.h5ad'
        self._cache_path = cache_dir / self._cache_path

    @property
    def dataset(self) -> AnnData:
        dataset = self._cached_dataset()
        if dataset is not None:
            return dataset

        dataset = self._read_raw_dataset()
        dataset = self._fill_annotations(dataset)
        if self._preprocess:
            dataset = self._preprocess_data(dataset)
        dataset.var_names_make_unique(join='_')

        dataset.write_h5ad(self._cache_path)

        return dataset
    
    @property
    def cell_treatment_map(self) -> pd.DataFrame:
        sars_guide_map = pd.read_csv(self._data_path_sars / 'crispr_analysis/protospacer_calls_per_cell.csv', index_col='cell_barcode')
        mock_guide_map = pd.read_csv(self._data_path_mock / 'crispr_analysis/protospacer_calls_per_cell.csv', index_col='cell_barcode')
        cell_treatment_map = pd.concat([sars_guide_map, mock_guide_map])
        cell_treatment_map.loc[:,'feature_call'] = cell_treatment_map.feature_call.str.replace(r'-\d+', '', regex=True)
        return cell_treatment_map

    def _cached_dataset(self):
        if self._ignore_cache:
            return None
        if not self._cache_path.exists():
            return None
        dataset = sc.read_h5ad(self._cache_path)
        return dataset

    def _read_raw_dataset(self):
#         sars_data_path = next(self._data_path_sars.glob('filtered_feature_bc_matrix'))
#         mock_data_path = next(self._data_path_mock.glob('filtered_feature_bc_matrix'))

#         sars_dataset = sc.read_10x_mtx(sars_data_path, cache=True)
#         mock_dataset = sc.read_10x_mtx(mock_data_path, cache=True)
#         dataset = sars_dataset.concatenate(mock_dataset, batch_categories=['SARS2', 'MOCK'], index_unique=None)       
        
        data_files = glob.glob(str(self._data_path / 'data/*/filtered_feature_bc_matrix'))
        sample_suffix = '_MON_crispr'

        # first load
        adatas = {}
        for i, file in enumerate(data_files):
            if i==0:
                adata = sc.read_10x_mtx(file, cache=True)
                batch_key = file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]
                adata.var_names_make_unique()
            else:
                adatas[file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]] = sc.read_10x_mtx(file, cache=True)
                adatas[file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]].var_names_make_unique()

        adata = adata.concatenate(*adatas.values(), batch_categories=[batch_key]+list(adatas.keys()), index_unique=None)
        del adatas
        # drop sars-cov-2 counts
        adata = adata[:, :-1]
        # drop suffixes from index
        adata.obs.index = adata.obs.index.str.replace('-SARS2', '').str.replace('-MOCK', '')
        # drop duplicate index
        adata = adata[~adata.obs.index.duplicated(keep=False)]
        return adata

    def _fill_annotations(self, dataset):
        dataset.obs['perturbation'] = np.nan
        treatment_map = self.cell_treatment_map
        for idx in dataset.obs.index:
            try:
                dataset.obs.loc[idx, 'perturbation'] = treatment_map.loc[idx, 'feature_call']
            except KeyError:
                pass
        return dataset

    def _metadata_from_perturbation_map(perturb_map):
        raise NotImplementedError

    def _preprocess_data(self, adata):
        sc.pp.calculate_qc_metrics(adata,inplace=True)
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['pmito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        print('Ncells=%d have >10percent mt expression' % np.sum(adata.obs['pmito']>0.1))
        print('Ncells=%d have <200 genes expressed' % np.sum(adata.obs['n_genes_by_counts']<200))
        print('Ngenes=%d have <3 genes expressed' % np.sum(adata.var['n_cells_by_counts'] < 3))
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3) # filtering cells gets rid of some genes of interest
        adata.raw = adata
        if False:
            # filter 
            adata = adata[adata.obs.pmito <= 0.1, :]

        # filter out cells from rare labels
        labels = adata.obs[['perturbation', 'batch']].value_counts()
        labels = labels[labels > 50]
        adata = adata[adata.obs.perturbation.isin(labels.loc[(slice(None), 'SARS2')].index)]

        sc.pp.normalize_total(adata)
        sc.pp.sqrt(adata)
        sc.tl.pca(adata, n_comps=100)       
        sc.external.pp.bbknn(adata, neighbors_within_batch=15, n_pcs=50)
        sc.tl.umap(adata)
        
#         adata.var['mito'] = adata.var_names.str.startswith('MT-')
#         sc.pp.calculate_qc_metrics(
#             adata, qc_vars=['mito'], percent_top=None, log1p=False, inplace=True)
#         print('Ncells=%d have >10percent mito expression' %
#             np.sum(adata.obs.pct_counts_mito > 10))
#         print('Ncells=%d have <200 genes expressed' %
#             np.sum(adata.obs.n_genes_by_counts < 200))
#         print('Ngenes=%d have <3 cells expressed' %
#             np.sum(adata.var.n_cells_by_counts < 3))
#         adata.raw = adata

#         # filter by cell/gene
#         sc.pp.filter_cells(adata, min_genes=200)
#         sc.pp.filter_genes(adata, min_cells=3)

#         # NOTE: Do not filter by mitochondrial expression levels as it makes the distributions closer.
#         # Think if that should be done or not.
#         # adata = adata[adata.obs.pct_counts_mito <= 10, :]

#         sc.pp.normalize_total(adata)
#         sc.pp.sqrt(adata)

#         sc.tl.pca(adata, n_comps=100)
#         sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
#         sc.tl.umap(adata)        
        return adata