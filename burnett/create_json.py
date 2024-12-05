#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 04 Dec 2024 16:39
# @Author: Yao LI
# @File: burnett/create_json.py
import os
import sys
import json
from utils import read_fasta, split_fasta

AF3_CHR = ['A', 'R', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'N', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def is_valid_string(input_string):
    """AlphaFold3 only allows certain sets of characters"""
    allowed_characters = set("ARDCQEGHILKMNFPSTWYV")
    allowed_set = set(allowed_characters)
    return all(char in allowed_set for char in input_string)


def parse_af3_fasta(fasta: dict):
    """Discard protein sequences containing invalid characters"""
    return {format_af3_name(key): value for key, value in fasta.items() if is_valid_string(value)}


def format_af3_name(job_id):
    """set delimiters to be underscores. AlphaFold3 does not allow dots in job name"""
    import re
    new_job_id = re.sub("[^A-Za-z0-9]+", "_", job_id)
    return new_job_id


def af3_entry(k, v):
    """
    Create one entry in AlphaFold3 format
    :param k: job id
    :param v: protein sequence
    :return:
    """
    out_dict = {
        "name": k,
        "modelSeeds": [1],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": v,
                    "count": 1
                }
            }
        ]
    }
    return out_dict


def parse_af3_json(fasta: dict):
    """
    Create a list of entries in AlphaFold3 format
    :param fasta:
    :return:
    """
    total = []
    for k, v in fasta.items():
        total.append(af3_entry(k, v))
    return total


if __name__ == '__main__':
    database_dir = '/home/share/huadjyin/home/fanguangyi/liyao1/04.predict_structures/exp/6.alphafold_second_round'
    species = sys.argv[1]
    fn = os.path.join(database_dir, f'sub_fa/{species}_not_predicted_esmfold2.fa')
    # 1. read fasta file
    fasta = read_fasta(fn, max_len=5000, format=False)
    # ensure the format of the fasta entries meet the AlphdFold3 requirement
    fasta = parse_af3_fasta(fasta)
    # 2. Split entries into chunks
    sub_fasta_list = split_fasta(fasta, chunk_size=20)
    # 3. Save spliced fasta into json files, each file contains {chunk_size} entries
    n = 1
    for sub_fasta in sub_fasta_list:
        entry_chunk = parse_af3_json(sub_fasta)  # format fasta into a list of dictionaries
        with open(f'protein_jsons/{species}/{species}_{n}.json', 'w') as f:
            json.dump(entry_chunk, f, indent=4)
        n += 1
