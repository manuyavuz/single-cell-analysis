import anndata
from scanpy.readwrite import read
from scanpy import AnnData
from scanpy import settings
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

class SciPlex():
    def __init__(self, data_path, preprocess=False, ignore_cache=False, cache_dir=None):
        self._data_path = Path(data_path)
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
        dataset, treatment_map, cell_treatment_map = self._read_raw_dataset()
        dataset = self._fill_annotations(dataset, cell_treatment_map)
        dataset.var_names_make_unique(join='_')

        if self._preprocess:
            dataset = self._preprocess_data(dataset)

        dataset.write_h5ad(self._cache_path)

        return dataset

    def _cached_dataset(self):
        if self._ignore_cache:
            return None
        if not self._cache_path.exists():
            return None
        dataset = sc.read_h5ad(self._cache_path)
        return dataset

    def _read_raw_dataset(self):
        data_path = self._data_path

        matrix_path = next(data_path.glob('*UMI.count.matrix'))
        cells_path = next(data_path.glob('*cell.annotations.txt'))
        genes_path = next(data_path.glob('*gene.annotations.txt'))
        hash_sheet_path = next(data_path.glob('*hashSampleSheet.txt'))
        hash_table_out_path = next(data_path.glob('*hashTable.out.txt'))

        cells = pd.read_table(
            cells_path,
            delim_whitespace=True,
            usecols=[0],
            index_col=0,
            names=[None])

        genes = pd.read_table(
            genes_path,
            delim_whitespace=True,
            names=['gene_ids', 'gene_symbols'],
            index_col='gene_symbols')
        genes.index.name = None

        treatment_map = pd.read_table(
            hash_sheet_path,
            delim_whitespace=True,
            names=['treatment', 'umi', 'umi_count'])

        cell_treatment_map = pd.read_table(
            hash_table_out_path,
            delim_whitespace=True,
            names=['sample', 'barcode', 'treatment', 'axis', 'umi_count'])

        matrix = sc.read_mtx(matrix_path)
        dataset = matrix.T

        dataset.obs = cells
        dataset.var = genes

        return dataset, treatment_map, cell_treatment_map

    def _fill_annotations(self, dataset, cell_treatment_map):
        cells = []
        metadata = []
        for name, group in cell_treatment_map[cell_treatment_map.barcode.isin(dataset.obs.index)].groupby(['barcode']):
            cells.append(name)
            freq_treatment = group.iloc[group.umi_count.argmax()]
            perturb_map = freq_treatment["treatment"].split('_')
            metadata.append(self._metadata_from_perturbation_map(perturb_map))

        dataset = dataset[dataset.obs_names.isin(cells)].copy()

        metadata = pd.DataFrame(metadata, index=dataset.obs_names)
        dataset.obs = pd.concat([dataset.obs, metadata], axis=1)
        return dataset

    def _metadata_from_perturbation_map(perturb_map):
        raise NotImplementedError

    def _preprocess_data(self, adata):
        adata.var['mito'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['mito'], percent_top=None, log1p=False, inplace=True)
        print('Ncells=%d have >10percent mito expression' %
            np.sum(adata.obs.pct_counts_mito > 10))
        print('Ncells=%d have <200 genes expressed' %
            np.sum(adata.obs.n_genes_by_counts < 200))
        print('Ngenes=%d have <3 genes expressed' %
            np.sum(adata.var.n_cells_by_counts < 3))
        adata.raw = adata

        # filter by cell/gene/mito expression values
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.pct_counts_mito <= 10, :]

        sc.pp.normalize_total(adata)
        sc.pp.sqrt(adata)

        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        return adata


class SciPlex2(SciPlex):
    def _metadata_from_perturbation_map(self, perturb_map):
        metadata = {
            'perturbation_raw': perturb_map[:-1],
            'cell_line': 'A549',
            'treatment': perturb_map[0],
            'dose': float(perturb_map[1]),
        }
        metadata[perturb_map[0]] = float(perturb_map[1])
        return metadata

class SciPlex4(SciPlex):
    def _metadata_from_perturbation_map(self, perturb_map):
        metadata = {
            'perturbation_raw': perturb_map[:-1],
            'cell_line': perturb_map[-3],
        }
        for t, d in zip(perturb_map[1:-3:2], perturb_map[0:-3:2]):
            metadata[t] = float(d)
        return metadata
