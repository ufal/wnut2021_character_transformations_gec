#!/usr/bin/env python3
import gzip
from typing import List
import unicodedata

import Levenshtein

class Aligner:
    def _lcs(self, s, c):
        lcs = [[0] * (len(c) + 1) for _ in range(len(s) + 1)]
        step = [[0] * (len(c) + 1) for _ in range(len(s) + 1)]
        for i in reversed(range(len(s))):
            for j in reversed(range(len(c))):
                lcs[i][j], step[i][j] = lcs[i+1][j], 0
                for l in range(1, len(c)-j + 1):
                    if l > 3 * len(s[i]) + 8:
                        break
                    if c[j:j + l].isspace():
                        continue
                    if s[i] == c[j:j + l]:
                        weight = 1
                    elif s[i].strip() == c[j:j + l].strip():
                        weight = 0.75
                    else:
                        weight = Levenshtein.ratio(s[i], c[j:j + l]) * 0.5

                    if weight + lcs[i + 1][j + l] > lcs[i][j]:
                        lcs[i][j], step[i][j] = weight + lcs[i + 1][j + l], l
        return lcs, step

    def _rewrite_for_matching(self, s):
        return "".join(
            "." if c in ",.!?;" else unicodedata.normalize("NFD", c)[0].lower() for c in s
        )

    def _best_alignment(self, s, c):
        lcs, step = self._lcs(list(map(self._rewrite_for_matching, s)), self._rewrite_for_matching(c))
        alignment = []

        j = 0
        for i in range(len(s)):
            l = len(c) - j if i + 1 == len(s) else step[i][j]
            alignment.append(c[j:j + l])
            j += l

        return alignment

    def align(self, subwords:List[str], chars:str) -> List[str]:
        return self._best_alignment(subwords, chars)

if __name__ == "__main__":
    import argparse
    import multiprocessing
    import pickle
    import sys

    import tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, help="Input data")
    parser.add_argument("--dump", default=None, type=str, help="Dump alignment to")
    parser.add_argument("--first_correct", default=0, type=int, help="First variant is correct")
    parser.add_argument("--limit", default=0, type=int, help="Input data limit")
    parser.add_argument("--threads", default=16, type=int, help="Threads to use")
    parser.add_argument("--tokenizer", default="electra", type=str, help="Tokenizer to use")
    args = parser.parse_args()

    aligner = Aligner()
    tokenizer = tokenizer.Tokenizer.create(args.tokenizer)
    input_subwords, gold_strings = [], []

    with open(args.data, "r", encoding="utf-8") as data_file:
        for line in data_file:
            if args.limit and len(input_subwords) >= args.limit:
                break

            columns = line.rstrip("\n").split("\t")
            if args.first_correct:
                if len(columns) > 2:
                    print("When first variant is correct, only two columns can be present, but saw {}, ignoring".format(len(columns)))
                    continue
                input_line, gold_lines = columns[1], columns[:1]
            else:
                input_line, gold_lines = columns[0], columns[1:]

            assert len(gold_lines)
            input_tokenized = tokenizer.apply(input_line)

            if len(input_tokenized) > 5000:
                print("Skipping sentence with {} tokens".format(len(input_tokenized)), file=sys.stderr, flush=True)
                continue

            for gold_line in gold_lines:
                input_subwords.append(input_tokenized)
                gold_strings.append(tokenizer.process_input(gold_line))

    inputs = list(zip(input_subwords, gold_strings))

    def perform_align(inputs):
        return inputs[0], aligner.align(*inputs)

    pool, results = multiprocessing.Pool(args.threads), []
    for subwords, alignment in pool.imap(perform_align, inputs):
        results.append((subwords, alignment))
        if len(results) % 1000 == 0:
            print(len(results), file=sys.stderr, flush=True)
            sys.stdout.flush()

    with gzip.open(args.dump, "wb") as dump_file:
        pickle.dump(results, dump_file)
