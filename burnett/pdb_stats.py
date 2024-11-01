#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Oct 2024 10:14
# @Author: Yao LI
# @File: burnett/pdb_stats.py
import os
import sys

import glob
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_struc_file(fn):
    """read pdb file"""
    # load protein structure file (*.pdb file)
    import biotite.structure.io as bsio
    struct = bsio.load_structure(fn, extra_fields=['b_factor'])
    return struct


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

aa_3_N = {a: n for n, a in enumerate(alpha_3)}
aa_N_1 = {n: a for n, a in enumerate(alpha_1)}


def read_pdb(fn):
    colspecs = [(0, 4), (4, 11), (12, 17), (17, 20), (21, 27),
                (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)]
    df = pd.read_fwf(fn, header=None, colspecs=colspecs, skipfooter=2)  # skiprows=[1],
    return df


def mean_ppdlt(data):
    m_plddt = data[9].mean()
    return m_plddt


def proportion_ppdlt(data, cutoff=70):
    plddt = data[9]
    perct = len(plddt[plddt >= cutoff]) / len(plddt)
    return perct


def species_pdb_stats(species_dir):
    species_pdbs = glob.glob(f'{species_dir}/*.pdb')
    total_mean_plddt = {}  # plddt mean values
    total_percet_plddt = {} # num of plddt values larger than cutoff / total plddt nums
    for fn in species_pdbs:
        struct = read_pdb(fn)
        pid = os.path.basename(fn).strip('.pdb')
        total_mean_plddt[pid] = mean_ppdlt(struct)
        total_percet_plddt[pid] = proportion_ppdlt(struct)
    return total_percet_plddt, total_mean_plddt


if __name__ == '__main__':
    species_dir = sys.argv[1]
    p_list, m_list = species_pdb_stats(species_dir)

    list_50 = [k for k,v in p_list.items() if v >= 0.5]  # > 50 percent plddt is larger than 70
    list_70 = [k for k,v in m_list.items() if v >= 70]  # mean plddt value is larger than 70
    print(f'{len(list_50)} out of {len(p_list)} proteins have more than 50% atom which plddt value is bigger than 70.')
    print(f'{len(list_70)} out of {len(m_list)} proteins have mean plddt value that is larger than 70.')
    with open(f'{species_dir}_low_plddt_percent.txt', 'w') as f:
        f.writelines('\n'.join(list_50))
    with open(f'{species_dir}_low_plddt_mean.txt', 'w') as f:
        f.writelines('\n'.join(list_70))

    # plot histogram
    plt.hist(list(p_list.values()))
    plt.xlabel('percentage of pLDDT greater than 70')
    plt.ylabel('num of proteins')
    plt.title(f'{species_dir}')
    plt.savefig(f'{species_dir}_percent.png')
    plt.close()

    plt.hist(list(m_list.values()))
    plt.xlabel('mean pLDDT value')
    plt.ylabel('num of proteins')
    plt.title(f'{species_dir}')
    plt.savefig(f'{species_dir}_mean.png')
    plt.close()
