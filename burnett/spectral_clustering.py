#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 23 May 2024 09:47
# @Author: Yao LI
# @File: burnett/spectral_clustering.py
import sys
import os

import argparse
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding


def get_module(module_path, module_name):
    """
    import a class from a Python file using an absolute path
    :param module_path: path to the module.py file
    :param module_name: name of the module.py file
    :return:
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


saturn_model_path = '/home/share/huadjyin/home/fanguangyi/liyao1/tools/SATURN/model/saturn_model.py'
saturn_model = get_module(saturn_model_path, "saturn_model")


class ProteinStuctureEmbeddings(saturn_model.SATURNPretrainModel):
    def __init__(self, species_order=None, alignment_fn=None, map_dir=None):
        self.species_info = {}

        self.species_order = species_order

        self.map_dir = map_dir
        self.pid_mappings = {}

        self.alignment_fn = alignment_fn
        self.simi_matrix_fn = ''
        self.simi_matrix = None

        self.labels = None
        self.structure_embeddings = None
        self.protein_cluster = None

    def read_mapping(self, species_name: str) -> pd.DataFrame:
        """
        Read protein name to database protein id mapping file
        :param map_dir: directory that stores protein-id mapping files
        :param species_name: target species name
        :return:
        """
        fn = os.path.join(self.map_dir, f'{species_name}_pid_protein_name.csv')
        mapping = pd.read_csv(fn, sep='\t')
        mapping = mapping.astype(str)
        mapping['gene_name'] = mapping['symbol'].str.split(' ', expand=True)[0]
        mapping['symbol'] = mapping['gene_name'] + f'_{species_name}'
        mapping['pid'] = mapping['pid'].str.lower()
        self.pid_mappings[species_name] = mapping
        return mapping

    def get_species_gene_index(self):
        """
        Generate a dictionary, gene start-end index for the species
        :return:
        """
        sorted_species_order = sorted(self.species_order)
        n = 0
        for species in sorted_species_order:
            self.species_info[species] = {}
            self.species_info[species]['gene_index'] = ()
            self.species_info[species]['gene_names'] = []
            self.species_info[species]['pid'] = []

            mapping_df = self.read_mapping(species)
            gene_names = list(mapping_df.gene_name)
            # self.total_gene_names[species] = gene_names
            # self.species_gene_index[species] = (n, n + len(gene_names))
            self.species_info[species]['gene_names'] = gene_names
            self.species_info[species]['gene_index'] = (n, n + len(gene_names))
            self.species_info[species]['pid'] = mapping_df.pid
            self.species_info[species]['mapping'] = mapping_df
            n += n + len(gene_names)
            # print(f'{species} has {len(gene_names)} protein structures aligned.')
        return self.species_info

    def get_simi_matrix(self):  # , species_name):#mapping: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Read protein-protein structure alignment output, TM score indicating structure similarity
        :param fn: alignment file. USalign output
        :param mapping: [Optional] protein-id mapping dataframe
        :return:
        """
        df = pd.read_csv(self.alignment_fn, sep='\t', dtype=str)
        print(f'total {len(set(df.PDBchain1))+1} proteins in alignment file.')
        df.PDBchain2 = df.PDBchain2.str.split('.', expand=True)[0]
        df.TM1 = df.TM1.astype('float64')
        df.TM2 = df.TM2.astype('float64')
        # choose minimum value
        df['small_TM'] = df[['TM1', 'TM2']].min(axis=1)
        df = df[['PDBchain1', 'PDBchain2', 'small_TM']]

        # Format similarity matrix
        simi_matrix = df.pivot(index='PDBchain1', columns='PDBchain2', values='small_TM')
        # Make matrix symmetric
        simi_matrix = simi_matrix.combine_first(simi_matrix.T)
        simi_matrix = simi_matrix.fillna(1)

        # Map each protein name to species-gene name
        # if mapping:
        # TODO: 还需要这一步吗
        # if self.pid_mappings[species_name]:
        #     simi_matrix.rename(columns=dict(zip(mapping.pid, mapping.symbol)),
        #                        index=dict(zip(mapping.pid, mapping.symbol)),
        #                        inplace=True)
        self.simi_matrix = simi_matrix
        print(simi_matrix)
        return simi_matrix

    def clustering_embedding(self, n_clusters: int = 20):
        """

        :param simi_matrix:
        :param output_dir:
        :param n_clusters:
        :return:
        """
        proteins_order = list(self.simi_matrix.index)
        # Perform spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(self.simi_matrix.to_numpy())
        labels = clustering.labels_
        self.labels = labels
        self.protein_cluster = pd.DataFrame({
            'pid': proteins_order,
            'label': labels.ravel(),
        })
        # self.protein_cluster.to_csv(os.path.join(output_dir, 'protein_cluster.csv'), index=False, sep='\t')

        # Create protein embeddings
        maps = spectral_embedding(
            clustering.affinity_matrix_,
            n_components=clustering.n_clusters,
            eigen_solver=clustering.eigen_solver,
            random_state=0,
            eigen_tol=clustering.eigen_tol,
            drop_first=False,
        )
        # np.savetxt(os.path.join(output_dir, 'maps.txt'), maps)
        self.structure_embeddings = maps
        return clustering, maps

    def get_species_structure_embeddings(self, species_name, output_dir='.')->np.array:
        """
        Some
        :param species_name:
        :param output_dir:
        :return:
        """
        species_pids = list(set(self.species_info[species_name]['pid']))
        # find which protein were clustered
        common_cluster_results = self.protein_cluster[self.protein_cluster.pid.isin(species_pids)]
        maps_array_index = common_cluster_results.index
        # find corresponding genes of clustered proteins
        used_pids = list(common_cluster_results['pid'])
        used_genes = list(self.species_info[species_name]['mapping'][self.species_info[species_name]['mapping']['pid'].isin(used_pids)].drop_duplicates(subset=['pid'])['gene_name'])
        print(f'{species_name}: used_genes: {len(used_genes)}')
        self.species_info[species_name]['clustered_genes'] = used_genes
        # get embeddings array
        species_maps = self.structure_embeddings[maps_array_index, :]
        print(f'{species_name} has {species_maps.shape[0]} proteins generated by spectral clustering.')
        np.savetxt(os.path.join(output_dir, f'{species_name}_maps.txt'), species_maps)
        return species_maps

    def get_structure_embeddings(self, species_name, data_genes)->np.array:
        """
        Some
        :param species_name:
        :param data_genes:
        :return:
        """
        # Given data gene names, find out its corresponding pids. Note that many of the genes don't have a pid
        mapping_df = self.species_info[species_name]['mapping']
        # 1. find data genes that has protein structures
        gene_has_proteins = set(data_genes).intersection(set(mapping_df.gene_name))
        sub_df = mapping_df[mapping_df.gene_name.isin(list(gene_has_proteins))].drop_duplicates(subset=['pid'])
        self.species_info[species_name]['used_genes'] = sub_df.gene_name
        gene_has_proteins_pids = sub_df.pid
        # 2. find protein structures that had been clustered
        clustered_pids = set(gene_has_proteins_pids).intersection(self.protein_cluster.pid)
        clustered_pids_index = self.protein_cluster[self.protein_cluster.pid.isin(list(clustered_pids))].index
        # sub_df = mapping_df[mapping_df.pid.isin(list(clustered_pids))]
        # clustered_gene_names = set(sub_df.gene_name).intersection(set(data_genes))
        # self.species_info[species_name]['clustered_genes'] = clustered_gene_names
        print(f'{len(data_genes)} input genes, found {len(clustered_pids)} aligned proteins.')
        target_embeddings_array = self.structure_embeddings[clustered_pids_index, :]
        return target_embeddings_array

    def evaluate(self):
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        # Evaluate clustering
        sil_score = silhouette_score(self.simi_matrix, self.labels, metric='precomputed')
        db_index = davies_bouldin_score(self.simi_matrix, self.labels)
        print(f'Silhouette Score: {sil_score}')
        print(f'Davies-Bouldin Index: {db_index}')

    def parse_input_table(self, fn):
        df = pd.read_csv(fn)
        sorted_species_order = sorted(self.species_order)
        for species_name in sorted_species_order:
            adata_fn = df[df.species == species_name]['path']


def get_pse(species_order, alignment_fn, mapping_results_dir):
    pse = ProteinStuctureEmbeddings(species_order, alignment_fn, map_dir=mapping_results_dir)
    pse.get_species_gene_index()
    print('Parse similarity matrix from alignment file')
    total_simi_matrix = pse.get_simi_matrix()
    print('Spectral clustering on protein structure similarity matrix')
    clustering, total_structure_embeddings = pse.clustering_embedding(n_clusters=30)
    print('Subset spectral clustering results by species')
    protein_embeddings = []
    protein_structure_array = pse.get_species_structure_embeddings(species_name)
    print(protein_structure_array.shape)
    return pse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alignment', type=str, help='alignment all file')
    parser.add_argument('-o', '--output', type=str, help='Output saving directory')
    parser.add_argument('-m', '--mapping', type=str, default=None,
                        help='[Optional] Directory stores mapping files of gene id/name and protein database entry id')
    parser.add_argument('-s', '--species', type=str, default=None, help='[Optional] Species name if applicable')
    args = parser.parse_args()

    # if args.mapping and args.species:
    #     mapping = read_mapping(args.mapping, args.species)
    # # Load USAlign protein structure alignment results
    # simi_matrix = get_simi_matrix(fn, mapping=None)
    # # Perform spectral clustering
    # clustering, maps = clustering_embedding(simi_matrix, output_dir)
    # evaluate(maps, clustering.labels_)

    # 2024-07-05
    # results to each species
    alignment_fn = '/home/share/huadjyin/home/fanguangyi/liyao1/01.evo_inter/data/fish_cifs/alignments/alignment.txt'
    mapping_results_dir = '/home/share/huadjyin/home/fanguangyi/liyao1/01.evo_inter/exp/rscb_results'
    species_order = ['indian_medaka', 'lungfish', 'whitespotted_bambooshark', 'zebrafish', 'gray_bichir',
                     'tarpon']  # 'human'

    species_name = species_order[3]
    print(species_name)
    pse = ProteinStuctureEmbeddings(species_order, alignment_fn, map_dir=mapping_results_dir)
    pse.get_species_gene_index()
    print('Parse similarity matrix from alignment file')
    total_simi_matrix = pse.get_simi_matrix()
    print('Spectral clustering on protein structure similarity matrix')
    total_structure_embeddings = pse.clustering_embedding(n_clusters=30)
    # maps = np.loadtxt('/home/share/huadjyin/home/fanguangyi/liyao1/01.evo_inter/exp/03.six_fishes/maps.txt')
    # pse.protein_cluster = pd.read_csv('/home/share/huadjyin/home/fanguangyi/liyao1/01.evo_inter/exp/03.six_fishes/protein_cluster.csv', sep='\t')
    # pse.protein_cluster['pid'] = pse.protein_cluster['symbol']
    # pse.structure_embeddings = maps
    print('Subset spectral clustering results by species')

    # fn = '/home/share/huadjyin/home/fanguangyi/guolidong/ProjectFishDigestive/intestine/00.data/Ome.h5ad'
    fn = '/home/share/huadjyin/home/fanguangyi/guolidong/ProjectFishDigestive/intestine/00.data/Cpl.h5ad'
    import scanpy as sc
    adata = sc.read_h5ad(fn)
    protein_structure_array = pse.get_structure_embeddings(species_name, list(adata.var_names))
    print(protein_structure_array.shape)
