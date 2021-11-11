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
    parser.add_argument("--data", type=str, help="Data to encode")
    parser.add_argument("--dump", type=str, help="Dump data to")
    parser.add_argument("--rules", type=str, help="Rules to use")
    parser.add_argument("--rules_file", type=str, help="File with the rules")
    parser.add_argument("--others", default=False, action="store_true", help="Search for other rules")
    parser.add_argument("--resolve_identical", default=False, action="store_true", help="Resolve even identical subwords")
    parser.add_argument("--threads", default=16, type=int, help="Threads to use")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    random.seed(42)

    # Load the rules
    rules_dict = {}
    with open(args.rules_file, "r", encoding="utf-8") as rules_file:
        for line in rules_file:
            line = line.rstrip("\n")
            rules_dict[line] = (rules.Rule.from_str(line), len(rules_dict))

    # Load the data
    with gzip.open(args.data, "rb") as data_file:
        data = pickle.load(data_file)

    rule_factory = rules.Rule.create(args.rules)
    tokenizer = tokenizer.Tokenizer.create(args.tokenizer)

    def process_sentence(data):
        subwords, alignment = data
        total, changed, changed_matched, subword_changed, subword_changed_matched, results = 0, 0, 0, 0, 0, []
        for i in range(len(subwords)):
            if args.resolve_identical or subwords[i] != alignment[i]:
                subword_changed += 1

                default_rule = rule_factory(subwords, alignment, i).to_str()
                if default_rule in rules_dict:
                    results.append(rules_dict[default_rule][1])
                    subword_changed_matched += 1
                else:
                    results.append(1)
                    if args.others:
                        shuffled_rules = list(rules_dict.items())
                        random.shuffle(shuffled_rules)
                        for rule_str, (rule, rule_id) in shuffled_rules:
                            if rule.apply(subwords, i) == alignment[i]:
                                results[-1] = rule_id
                                subword_changed_matched += 1
                                break
            else:
                results.append(0)

            if i + 1 == len(subwords) or subwords[i + 1].startswith(' '):
                total += 1
                changed += 1 if subword_changed else 0
                changed_matched += 1 if subword_changed and subword_changed_matched == subword_changed else 0
                subword_changed, subword_changed_matched = 0, 0

        return total, changed, changed_matched, np.stack([tokenizer.tokens_to_ids(subwords), results], axis=1).astype(np.int32)

    total, changed, changed_matched, results = 0, 0, 0, []
    with multiprocessing.Pool(args.threads) as pool:
        for s_total, s_changed, s_changed_matched, s_results in pool.imap(process_sentence, data, args.threads):
            total += s_total
            changed += s_changed
            changed_matched += s_changed_matched
            results.append(s_results)

    print("Mapping: {:5.2f} {:5.2f}".format(
        100 * changed_matched / changed,
        100 * (total - changed + changed_matched) / total,
    ), file=sys.stderr)

    with gzip.open(args.dump, "wb") as dump_file:
        pickle.dump(results, dump_file)
