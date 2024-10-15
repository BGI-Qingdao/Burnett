#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Oct 2024 14:18
# @Author: Yao LI
# @File: burnett/predict.py
import os
import torch
import esm
import argparse


def read_fasta(fasta_file, len_cutoff: int=0):
    sequences = {}
    with open(fasta_file, "r") as f:
        current_sequence_id = None
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # header line
                if current_sequence_id and len(current_sequence) >= len_cutoff:  # check if length > than cutoff
                    sequences[current_sequence_id] = current_sequence
                current_sequence_id = line[1:]
                current_sequence = ""
            else:  # sequence line
                current_sequence += line
        # save last sequence (if meet length requirement)
        if current_sequence_id and len(current_sequence) >= len_cutoff:
            sequences[current_sequence_id] = current_sequence
    return sequences


def get_sequence(fasta):
    pass


def load_struc_file(fn):
    # load protein structure file (*.pdb file)
    import biotite.structure.io as bsio
    struct = bsio.load_structure(fn, extra_fields=['b_factor'])
    return struct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fa', type=str, help='protein sequence fa file')
    parser.add_argument('-o', '--output', type=str, help='path to output pdb directory')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='[Optional] file of list of gene names to predict if applicable. '
                             'Use all genes in the fa file when set to None')
    parser.add_argument('-s', '--species', type=str, default=None, help='[Optional] Species name if applicable')
    args = parser.parse_args()

    # 0. Load in pretrained models
    model_path = '/home/share/huadjyin/home/fanguangyi/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt'
    # For esm2 pretrained models
    # model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
    print('Loading pretrained model...')
    import esm.esmfold.v1.pretrained
    model = esm.esmfold.v1.pretrained._load_model(model_path)
    model = model.eval().cuda()
    # print(type(model))  # class 'esm.esmfold.v1.esmfold.ESMFold'

    # 1. Load in protome data for one species
    fasta = read_fasta(args.fa, len_cutoff=0)
    print('Load fasta file done.')

    # 2. get sequences
    pid, sequence = get_sequence()
    # pid = "protein1"
    # sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    # 3. predict protein structures
    print('Predicting...')
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    # print(type(output))  # string

    print('Saving...')
    with open(os.path.join(args.output, f'{pid}.pdb'), 'w') as f:
        f.write(output)
