#!/bin/bash

export TOKENIZERS_PARALLELISM=false

for tokenizer_name_model in bert_uncased:bert-base-multilingual-uncased bert_cased:bert-base-multilingual-cased robeczech:ufal/robeczech-base; do
  tokenizer_name=${tokenizer_name_model%%:*}
  tokenizer_model=${tokenizer_name_model#*:}

  for f in data_aligned/input/*/*.tsv; do
    case $tokenizer_name$f in
      robeczech*cs*) ;;
      robeczech*) continue;;
    esac

    case $tokenizer_name$f in
      enbert_*en*) ;;
      enbert*) continue;;
    esac

    target=data_aligned/$tokenizer_name/${f#data_aligned/input/}
    target=${target%.tsv}.pickle.gz

    [ -f $target ] && continue

    case $f in
      *wnut* | *kazitext*) first_correct=1;;
      *) first_correct=0;;
    esac

    limit=0
    case $f in
      *wnut* | *kazitext* | *wiki* | *lang8*) limit=15000;;
      *) limit=0;;
    esac

    mkdir -p $(dirname $target)
    qsub -q cpu-troja.q -v TOKENIZERS_PARALLELISM=false -l mem_free=110G,h_vmem=64G -pe smp 31 -N aligner -j y -o $target.log venv/bin/python aligner.py --data=$f --dump=$target --tokenizer=$tokenizer_model --first_correct=$first_correct --limit=$limit --threads=24
  done
done
