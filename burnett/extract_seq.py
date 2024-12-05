#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 04 Dec 2024 15:01
# @Author: Yao LI
# @File: burnett/extract_seq.py
import os
import sys

from utils import read_fasta, read_list, save_fasta

database_dir = '/home/share/huadjyin/home/fanguangyi/liyao1/04.predict_structures/data'
outbase_dir = '/home/share/huadjyin/home/fanguangyi/liyao1/04.predict_structures/exp/6.alphafold_second_round'
species = sys.argv[1]
print(species)
fn = os.path.join(database_dir, f'{species}.fa')
id_list_fn = os.path.join(outbase_dir, f'{species}_low_plddt_mean.txt')

# 1. load in species genome sequence
fasta = read_fasta(fn, format=True, species=species)
k_list = list(fasta.keys())
print(k_list[:3])
print(len(fasta))  # check the num of total sequences
# 2. load in the ids of sequences that were predicted by ESMFold2
id_list = read_list(id_list_fn, species)
id_list = list(set(id_list))
if species =='gray_bichir':
    id_list = ['_'.join(s.split('_')[:3]) for s in id_list]
print(len(id_list))
print(id_list[:5])  # check if the ids were loaded correctly
# 3. generated id:seq of those were not predicted by ESMFold2
not_id = set(fasta.keys())-set(id_list)
print(len(not_id))  # check the num
sub_fasta = {k: fasta[k] for k in not_id if k in fasta}
print(len(sub_fasta))
save_fasta(sub_fasta, os.path.join(outbase_dir, f'{species}_low_plddt.fa'))
