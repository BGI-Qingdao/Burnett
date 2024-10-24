#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 12 Oct 2024 14:18
# @Author: Yao LI
# @File: burnett/predict.py
import os
import sys
import torch
# print(torch.__version__)  # 1.11.0+cu113
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import argparse
import glob


def format_pid_space(fasta_header, species=None):
    """replace space, comma or other symbols to underscore in pid"""
    import re
    if species == 'freshwater_butterflyfish':
        fasta_header = fasta_header.split(' ')[0]
    new_header = re.sub("[^A-Za-z0-9.]+", "_", fasta_header.strip(']'))
    return new_header


def get_gene_id(fasta_header, species):
    """Extract gene name/id from a fasta header"""
    pass


def read_fasta(fasta_file, len_cutoff=0):
    """
    Read fa/fasta file
    :param fasta_file: protome fa file name
    :param len_cutoff: amino acid sequence length cutoff (only seq longer than the len_cutoff will be kept)
    :return:
    """
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


def get_sequence(gene, fasta):
    """
    Retrieve amino acid sequence given a gene name
    :param gene: protein/gene name
    :param fasta: protome fasta dictionary
    :return:
    """
    seq = ''
    for k, v in fasta.items():
        if gene in k:
            seq = v
    return gene, seq


def load_struc_file(fn):
    """read pdb file"""
    # load protein structure file (*.pdb file)
    import biotite.structure.io as bsio
    struct = bsio.load_structure(fn, extra_fields=['b_factor'])
    return struct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict protein structure via ESMFold",
                                     fromfile_prefix_chars="@",
                                     add_help=True,
                                     epilog="Example: python predict.py -f test.fa -o /proj/results/ -s mouse --length 50"
                                     )
    parser.add_argument('-f', '--fa', type=str, help='protein sequence fa file')
    parser.add_argument('-o', '--output', type=str, help='path to output pdb directory')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='[Optional] file of list of gene names to predict if applicable. '
                             'Use all genes in the fa file when set to None')
    parser.add_argument('-s', '--species', type=str, default=None, help='[Optional] Species name if applicable')
    parser.add_argument('--length', type=int, default=0,
                        help='[Optional] minimal length of a amino acid sequence to use')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # create output directory
    if args.species:
        out_dir = os.path.join(args.output, args.species)
    else:
        out_dir = args.output
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # file saved
    file_list = glob.glob(f'{out_dir}/*.pdb')
    protein_predicted = [os.path.basename(i).strip('.pdb') for i in file_list]

    # 0. Load in pretrained models
    model_path = '/home/share/huadjyin/home/fanguangyi/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt'
    # For esm2 pretrained models
    # model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
    print('Loading pretrained model...')
    import esm.esmfold.v1.pretrained

    model = esm.esmfold.v1.pretrained._load_model(model_path)
    model = model.eval().cuda()  # class 'esm.esmfold.v1.esmfold.ESMFold'

    # 1. Load in protome data for one species
    fasta = read_fasta(args.fa, len_cutoff=args.length)  # dictionary
    print(f'Load fasta file done, contain {len(fasta)} entries.')

    # 2. get sequences
    if args.input:  # when user designate target genes/proteins to predict
        with open(args.input, 'r') as f:
            gene_list = f.read().splitlines()
        for gene in gene_list:
            pid, sequence = get_sequence(gene, fasta)
            # 3. predict protein structures
            print('Predicting...')
            with torch.no_grad():
                output = model.infer_pdb(sequence)  # string
            print('Saving...')
            with open(os.path.join(out_dir, '{}.pdb'.format(pid)), 'w') as f:
                f.write(output)
    else:  # when input gene_list file does not exist, use all genes/proteins in the fasta file
        n = 0
        x = 0
        for pid, sequence in fasta.items():
            reformated_pid = format_pid_space(pid)
            if reformated_pid in protein_predicted:  # if the protein was predicted already
                print('protein was predicted')
                n += 1
                continue
            else:
                # 3. predict protein structures
                print('Predicting...')
                x += 1
                with torch.no_grad():
                    output = model.infer_pdb(sequence)  # string
                print('Saving...')
                with open(os.path.join(out_dir, '{}.pdb'.format(reformated_pid)), 'w') as f:
                    f.write(output)
        print(f'{n} proteins was predicted, {x} have not.')
