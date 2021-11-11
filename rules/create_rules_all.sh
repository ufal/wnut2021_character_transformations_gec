#!/bin/bash

for model in bert_cased bert_uncased enbert_cased; do
  case $model in
    bert_uncased) diacritics="+postdiacritics";;
    *) diacritics="";;
  esac

  for r in {,words+}{replace_append,editscript,postcasing$diacritics+editscript}; do
    for synthetic in 0 1 2 3 5 9; do
      for l in cs de en ru; do
        case $model$l in
          enbert_caseden) ;;
          enbert_cased*) continue;;
        esac
        for occ in 1 2 3 4 5; do
          target=data_aligned/$model/$l.$synthetic.$r.$occ.rules
          qsub -q cpu-troja.q $args -N rules -j y -o /dev/null venv/bin/python create_rules.py --train $(ls data_aligned/$model/$l/train*.pickle.gz | grep -v .[0-9][0-9].) $(ls data_aligned/$model/$l/train*.00.pickle.gz | grep -v official | sed "s/$/:${synthetic}000/") --rules=$r --occurrences=$occ --dump=$target
        done
      done
    done
  done
done
