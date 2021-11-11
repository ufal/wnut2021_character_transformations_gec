#!/bin/bash

for f in data_encoded/*/*/*/test*.pickle.gz; do
  d=$(dirname $f)
  target=${f%.pickle.gz}.decoded
  echo $f
  [ -f $target ] || venv/bin/python apply_rules.py --copy_uncorrectable=1 --data=$f --rules_file $d/RULES.LIST* --tokenizer=$(cat $d/TOKENIZER) >$target
done
