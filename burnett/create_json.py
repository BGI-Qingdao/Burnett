#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 04 Dec 2024 16:39
# @Author: Yao LI
# @File: burnett/create_json.py
import os
import sys
import json
from utils import read_fasta, split_fasta


AF3_CHR = ['A','R','D','C','Q','E','G','H','I','L','K','M','N','F','P','S','T','W','Y','V']
def is_valid_string(input_string):
    """AlphaFold3 only allows certain sets of characters"""
    allowed_characters = set("ARDCQEGHILKMNFPSTWYV")
    allowed_set = set(allowed_characters)
    return all(char in allowed_set for char in input_string)

def format_af3_name(job_id):
    """set delimiters to be underscores. AlphaFold3 does not allow dots in job name"""
    import re
    new_job_id = re.sub("[^A-Za-z0-9]+", "_", job_id)
    return new_job_id


def parse_af3_fasta(fasta:dict):
    return {format_af3_name(key): value for key, value in fasta.items() if is_valid_string(value)}


def af3_entry(k, v):
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


def parse_af3_json(fasta:dict):
    total = []
    for k, v in fasta.items():
        total.append(af3_entry(k, v))
    return total


database_dir = '/home/share/huadjyin/home/fanguangyi/liyao1/04.predict_structures/exp/6.alphafold_second_round'
species = sys.argv[1]
fn = os.path.join(database_dir, f'{species}_low_plddt.fa')
fasta = read_fasta(fn, max_len=5000, format=False)
fasta = parse_af3_fasta(fasta)
sub_fasta_list = split_fasta(fasta, chunk_size=100)
n = 1
for sub_fasta in sub_fasta_list:
    entry_chunk = parse_af3_json(sub_fasta)
    with open(f'low_plddt_jsons/{species}/{species}_{n}.json', 'w') as f:
        json.dump(entry_chunk, f, indent=4)
    n += 1
