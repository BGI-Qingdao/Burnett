#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 15 Apr 2024 16:21
# @Author: Yao LI
# @File: Burnett/utils.py


def read_fasta(fasta_file: str, len_cutoff: int = 0) -> dict:
    """
    :param fasta_file:
    :param len_cutoff:
    :return:
    """
    sequences = {}
    with open(fasta_file, "r") as f:
        current_sequence_id = None
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # header line
                if current_sequence_id:  # check if there's any sequence read before
                    # check if the length of the current sequence meets the cutoff
                    if len(current_sequence) >= len_cutoff:
                        sequences[current_sequence_id] = current_sequence
                current_sequence_id = line[1:]
                current_sequence = ""
            else:  # sequence line
                current_sequence += line
        # save last sequence (if its length meets the cutoff)
        if current_sequence_id:
            if len(current_sequence) >= len_cutoff:
                sequences[current_sequence_id] = current_sequence
    return sequences


def save_fasta(fasta: dict, fn: str = 'new.fasta'):
    """

    :param fasta:
    :param fn:
    :return:
    """
    with open(fn, 'w') as f:
        for header, seq in fasta.items():
            f.writelines(f'>{header}\n{seq}\n')


def chunks(dir_data, chunk_size=100):
    from itertools import islice
    it = iter(dir_data)
    for i in range(0, len(dir_data), chunk_size):
        yield {k: dir_data[k] for k in islice(it, chunk_size)}


def split_fasta(fasta_fn, chunk_size=100):
    """split dictionary into chunks, each chunk contains chunk_size items"""
    fasta_dir = read_fasta(fasta_fn)
    n = 1
    for item in chunks(fasta_dir, chunk_size=chunk_size):
        save_fasta(item, fn=f'{n}.fa')
        n += 1


def search_by_gene_symbol(fasta: dict, gene_symbol: str):
    """
    case insensitive
    :param fasta:
    :param gene_symbol:
    :return:
    """
    matching_sequences = {}
    gene_symbol = gene_symbol.lower()  # Convert gene symbol to lowercase
    for key, value in fasta.items():
        if gene_symbol in key.lower():  # Convert key to lowercase for comparison
            matching_sequences[key] = value
    return matching_sequences


def jaccard_score(df1, df2, cluster_col1, cluster_col2, protein_col):
    import pandas as pd
    jaccard_scores = []
    # Iterate over each cluster in df1
    for cluster1 in df1[cluster_col1].unique():
        # Get proteins for the current cluster in df1
        proteins_df1 = set(df1[df1[cluster_col1] == cluster1][protein_col])
        # List to store similarity scores for current cluster
        cluster_similarities = []
        # Iterate over each cluster in df2
        for cluster2 in df2[cluster_col2].unique():
            # Get proteins for the current cluster in df2
            proteins_df2 = set(df2[df2[cluster_col2] == cluster2][protein_col])
            # Calculate Jaccard similarity
            intersection_size = len(proteins_df1.intersection(proteins_df2))
            union_size = len(proteins_df1.union(proteins_df2))
            jaccard_score_value = intersection_size / union_size if union_size != 0 else 0
            # Append similarity score to list
            cluster_similarities.append((cluster2, jaccard_score_value))
        # Sort cluster similarities based on similarity score
        sorted_cluster_similarities = sorted(cluster_similarities, key=lambda x: x[1], reverse=False)
        # Append all cluster similarities to jaccard_scores
        for cluster2, similarity_score in sorted_cluster_similarities:
            jaccard_scores.append([cluster1, cluster2, similarity_score])
    return pd.DataFrame(jaccard_scores, columns=['Cluster1', 'Cluster2', 'Jaccard Similarity'])


def jaccard_heatmap(jaccard_df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Pivot the DataFrame for plotting
    heatmap_data = pd.pivot_table(jaccard_df, values='Jaccard Similarity', index=['Cluster1'], columns=['Cluster2'], sort=False)#, dropna=False)
    heatmap_data = heatmap_data.iloc[:, ::-1]
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", cbar=True)
    plt.title('Jaccard Similarity Heatmap')
    plt.xlabel('Clusters')
    plt.ylabel('Clustering Output')
    plt.savefig('jaccard_heatmap.png')


def handle_cluster_output(cluster_output, ref):
    """

    :param cluster_output:
    :param ref:
    :return:
    """
    p1 = set(cluster_output.protein_ID)
    p2 = set(ref.protein_ID)
    non_p = p2 - p1
    non_df = pd.DataFrame({'cluster_number': [-1] * len(non_p), 'protein_ID': list(non_p)})
    cluster_output_with_non_cluster = pd.concat([cluster_output, non_df], ignore_index=True, sort=False)
    return cluster_output_with_non_cluster


def get_rest(ref_fn, downed_fn):
    with open(ref_fn, 'r') as f:
        l = f.readlines()
    ref_list = l[0].split(',')

    with open(downed_fn, 'r') as f:
        l = f.readlines()
    downed_list = [i.strip() for i in l]

    not_downed_list = list(set(ref_list) - set(downed_list))
    with open('pid_not_downed.txt', 'w') as f:
        f.writelines(','.join(not_downed_list))

