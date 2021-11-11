#!/usr/bin/env python3
import argparse
import gzip
import multiprocessing
import os
import pickle
import random
import sys

import numpy as np

import rules
import tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy_uncorrectable", default=1, type=int, help="Copy uncorrectable")
    parser.add_argument("--data", type=str, help="Data to apply rules to")
    parser.add_argument("--rules_file", type=str, help="File with the rules")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    # Load the rules
    rules_list = []
    with open(args.rules_file, "r", encoding="utf-8") as rules_file:
        for line in rules_file:
            line = line.rstrip("\n")
            rules_list.append(rules.Rule.from_str(line))

    # Load the data
    with gzip.open(args.data, "rb") as data_file:
        data = pickle.load(data_file)

    tokenizer = tokenizer.Tokenizer.create(args.tokenizer)

    for i, sentence in enumerate(data):
        subword_ids, rules = sentence[:, 0], sentence[:, 1]
        subwords = tokenizer.ids_to_tokens(subword_ids)
        result = []
        for i, rule in enumerate(rules):
            if rule == 1 and args.copy_uncorrectable:
                result.append(subwords[i])
            else:
                rule_result = rules_list[rule].apply(subwords, i)
                if rule_result is not None:
                    result.append(rule_result)
                else:
                    result.append(subwords[i])
        result = "".join(result)
        result = tokenizer.process_output(result).strip()

        print(result)
