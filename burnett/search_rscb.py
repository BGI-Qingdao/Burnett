#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 02 Apr 2024 15:48
# @Author: Yao LI
# @File: Burnett/search_rscb.py
import argparse
import requests
from rcsbsearchapi.search import SequenceQuery
import pandas as pd
import sys
from tqdm import tqdm


def search_database(amino_seq, evalue_cutoff=1, seq_identity: float = 0.9):
    """
    :param amino_seq:
    :param evalue_cutoff:
    :param seq_identity: sequence identity
    :return:
    """
    try:
        # Use SequenceQuery class and add parameters
        results = SequenceQuery(amino_seq, evalue_cutoff, seq_identity)
    except requests.exceptions.HTTPError:
        return None
    # results("polymer_entity") produces an iterator of IDs with return type - polymer entities
    for pid in results("entry"):
        return pid


def read_fasta(fasta_file, len_cutoff: int=0):
    sequences = {}  # 用于存储序列的字典
    with open(fasta_file, "r") as f:
        current_sequence_id = None
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # 标识行
                if current_sequence_id and len(current_sequence) >= len_cutoff:  # 添加条件检查序列长度是否大于等于70
                    sequences[current_sequence_id] = current_sequence
                current_sequence_id = line[1:]
                current_sequence = ""
            else:  # 序列行
                current_sequence += line
        # 存储最后一条序列（如果长度符合条件）
        if current_sequence_id and len(current_sequence) >= len_cutoff:
            sequences[current_sequence_id] = current_sequence
    return sequences


def main(species, fn, len_cutoff=100, evalue_cutoff=1, seq_identity=0.7):
    # 1. Load in protome data for one species
    fasta = read_fasta(fn, len_cutoff=len_cutoff)
    total_protein_num = len(fasta)
    print('Load fasta file done.')

    # 3. Search in PDB database
    top_pids = {}
    not_found = []
    print('Searching RSCB database...')
    n = 0
    # for name, amino_seq in tqdm(fasta.items(), total=total_protein_num, desc="Searching protein sequences in RSCB Database"):
    for name, amino_seq in fasta.items():
        n += 1
        if n % 100 == 0:
            print(f'{n}/{total_protein_num} proteins, found {len(top_pids)} matches')
        try:
            pid = search_database(amino_seq, evalue_cutoff=evalue_cutoff, seq_identity=seq_identity)
        except Exception:  # except requests.exceptions.HTTPError:
            pid = None
        # No match found
        if pid is None:
            not_found.append(name)
        # choose the best result
        else:
            top_pids[name] = pid

    print('Search RSCB database done.')

    # 3. save to file
    df = pd.DataFrame(top_pids.items(), columns=['symbol', 'pid'])
    df.to_csv(f'{species}_pid_protein_name.csv', sep='\t', index=False)
    # get_gene_name(df, species)

    with open(f'{species}_pid_list.txt', 'w') as f:
        f.writelines(','.join(list(top_pids.values())))

    with open(f'{species}_not_found_list.txt', 'w') as f:
        f.writelines('\n'.join(not_found))

    print('Saving to files done.')


def get_gene_name(df, species, col_name="symbol"):
    df['name'] = df[col_name].str.split(' ', expand=True)[2]
    sub = df[~df.name.str.startswith('gene_symbol')]
    names = list(sub.name)
    with open(f'{species}_not_found_names.txt', 'w') as f:
        f.writelines('\n'.join(names))
    return names


if __name__ == '__main__':
    print('Starting...')
    species_list = ['whitespotted_bambooshark', 'human', 'zebrafish', 'human', 'lungfish', 'tarpon', 'gray_bichir', 'indian_medaka']
    for s in species_list:
        print(f'Current species: {s}')
        fn = f'/home/share/huadjyin/home/fanguangyi/liyao1/evo_inter/data/protome/{s}.gene_symbol.fa'
        main(s, fn)
