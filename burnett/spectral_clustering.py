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


def read_mapping(map_dir: str, species_name: str) -> pd.DataFrame:
    """
    Read protein name to database protein id mapping file
    :param map_dir: directory that stores protein-id mapping files
    :param species_name: target species name
    :return:
    """
    fn = os.path.join(map_dir, f'{species_name}_pid_protein_name.csv')
    mapping = pd.read_csv(fn, sep='\t')
    mapping = mapping.astype(str)
    mapping['symbol'] = mapping['symbol'].str.split(' ', expand=True)[0]
    mapping['symbol'] = mapping['symbol'] + f'_{species_name}'
    print(mapping)
    return mapping


def read_alignments(fn: str, mapping: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Read protein-protein structure alignment output, TM score indicating structure similarity
    :param fn: alignment file. USalign output
    :param mapping: [Optional] protein-id mapping dataframe
    :return:
    """
    df = pd.read_csv(fn, sep='\t')
    # choose minimum value
    df['small_TM'] = df[['TM1', 'TM2']].min(axis=1)
    df = df[['PDBchain1', 'PDBchain2', 'small_TM']]
    # make PDBchain1 number equals to PDBchain2 number
    use_proteins = list(set(df.PDBchain1))
    df = df[df.PDBchain2.isin(use_proteins)]

    # Format similarity matrix
    simi_matrix = df.pivot(index='PDBchain1', columns='PDBchain2', values='small_TM')
    # Make matrix symmetric
    simi_matrix = simi_matrix.combine_first(simi_matrix.T)
    simi_matrix = simi_matrix.fillna(1)

    # Map each protein name to species-gene name
    if mapping:
        simi_matrix.rename(columns=dict(zip(mapping.pid, mapping.symbol)),
                           index=dict(zip(mapping.pid, mapping.symbol)),
                           inplace=True)

    print(simi_matrix)
    return simi_matrix


def clustering_embedding(simi_matrix: pd.DataFrame, output_dir: str, n_clusters: int = 20):
    proteins_order = list(simi_matrix.index)
    # Perform spectral clustering
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(simi_matrix.to_numpy())
    labels = clustering.labels_
    protein_cluster = pd.DataFrame({
        'symbol': proteins_order,
        'label': labels.ravel(),
    })
    protein_cluster.to_csv(os.path.join(output_dir, 'protein_cluster.csv'), index=False, sep='\t')

    # Create protein embeddings
    maps = spectral_embedding(
        clustering.affinity_matrix_,
        n_components=clustering.n_clusters,
        eigen_solver=clustering.eigen_solver,
        random_state=0,
        eigen_tol=clustering.eigen_tol,
        drop_first=False,
    )
    np.savetxt(os.path.join(output_dir, 'maps.txt'), maps)

    return clustering, maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alignment', type=str, help='alignment all file')
    parser.add_argument('-o', '--output', type=str, help='Output saving directory')
    parser.add_argument('-m', '--mapping', type=str, default=None, help='[Optional] Directory stores mapping files of gene id/name and protein database entry id')
    parser.add_argument('-s', '--species', type=str, default=None, help='[Optional] Species name if applicable')
    args = parser.parse_args()

    output_dir = args.output
    fn = args.alignment
    mapping = read_mapping(args.mapping, args.species)

    # Load USAlign protein structure alignment results
    simi_matrix = read_alignments(fn, mapping)

    # Perform spectral clustering
    clustering, maps = clustering_embedding(simi_matrix, output_dir)
