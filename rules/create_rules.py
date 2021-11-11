#!/usr/bin/env python3
import argparse
import collections
import gzip
import os
import pickle
import sys

import rules

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=str, help="Dump rules to file")
    parser.add_argument("--occurrences", type=int, default=1, help="Min occurrences")
    parser.add_argument("--rules", type=str, help="Rules to use")
    parser.add_argument("--train", type=str, nargs="+", help="Training alignments")
    args = parser.parse_args()

    # Load training data
    train = []
    for train_path in args.train:
        train_limit = None
        if ":" in train_path:
            train_path, train_limit = train_path.split(":")
            train_limit = int(train_limit)
        if train_limit != 0:
            with gzip.open(train_path, "rb") as train_file:
                train.extend(pickle.load(train_file)[:train_limit])

    # Create rules
    rule_factory = rules.Rule.create(args.rules)

    rule_dict = collections.defaultdict(lambda: 0)
    for subwords, alignment in train:
        for i in range(len(subwords)):
            rule_dict[rule_factory(subwords, alignment, i).to_str()] += 1

    # Special rules
    special_rules = [
        rule_factory(["KEEP"], ["KEEP"], 0).to_str(),
        rule_factory([""], ["[UNCORRECTABLE_ERROR]"], 0).to_str(),
    ]

    # Prune rules
    pruned_rules = sorted(rule for rule, count in rule_dict.items() if count >= args.occurrences and rule not in special_rules)

    # Final rules
    rules = special_rules + pruned_rules
    print("There are {} rules in total".format(len(rules)), file=sys.stderr)

    with open(args.dump, "w") as rules_file:
        for rule in rules:
            print(rule, file=rules_file)
