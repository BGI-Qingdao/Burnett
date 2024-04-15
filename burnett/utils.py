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

