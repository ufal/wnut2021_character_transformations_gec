#!/bin/bash

function generate {
  lang=$1
  tokenizer=$2
  exp=$3
  rules=$4

  target=data_encoded/$tokenizer/$lang/$exp
  [ -d $target ] && return

  mkdir -p $target
  raw_rules=${rules#*.}; raw_rules=${raw_rules%.*}; echo $raw_rules >$target/RULES
  echo ${rules##*.} >$target/RULES.OCCURRENCES
  cp data_aligned/$tokenizer/$lang.$rules.rules $target/RULES.LIST
  case $tokenizer in
    bert_uncased) full_tokenizer=bert-base-multilingual-uncased;;
    bert_cased) full_tokenizer=bert-base-multilingual-cased;;
    enbert_cased) full_tokenizer=bert-base-cased;;
    robeczech) full_tokenizer=ufal/robeczech-base;;
  esac
  echo $full_tokenizer >$target/TOKENIZER

  for data in data_aligned/$tokenizer/$lang/*.pickle.gz; do
    case $data in
      *.0[1-9].* | *.[1-9][0-9].*) p=-101;;
      *) p=-100;
    esac
    qsub -q cpu-troja.q -l mem_free=64G,h_data=64G -p $p -pe smp 25 -N rules -j y -o $target/$(basename $data .pickle.gz).log venv/bin/python encode_rules.py --data $data --dump $target/$(basename $data) --rules $(cat $target/RULES) --rules_file $target/RULES.LIST --others --threads=28 --tokenizer $(cat $target/TOKENIZER)
  done
}

#######################
# Subword-level rules #
#######################
generate cs bert_uncased 3_0k 0.postcasing+postdiacritics+editscript.4
generate cs bert_uncased 4_5k 1.postcasing+postdiacritics+editscript.3
generate cs bert_uncased 7_7k 1.postcasing+postdiacritics+editscript.2
generate cs bert_uncased large-23_9k 1.postcasing+postdiacritics+editscript.1

generate cs bert_cased 2_3k 0.postcasing+editscript.4
generate cs bert_cased 3_2k 0.postcasing+editscript.3
generate cs bert_cased 5_6k 0.postcasing+editscript.2
generate cs bert_cased large-18_8k 1.postcasing+editscript.1

generate cs robeczech 3_1k 0.editscript.4
generate cs robeczech 4_6k 2.editscript.3
generate cs robeczech 7_9k 1.editscript.2

generate de bert_uncased 1_5k 0.postcasing+postdiacritics+editscript.3
generate de bert_uncased 2_4k 0.postcasing+postdiacritics+editscript.2
generate de bert_uncased 4_3k 9.postcasing+postdiacritics+editscript.2
generate de bert_uncased large-12_0k 3.postcasing+postdiacritics+editscript.1

generate de bert_cased 1_0k 0.postcasing+editscript.4
generate de bert_cased 2_1k 0.postcasing+editscript.2
generate de bert_cased 3_5k 0.postcasing+editscript.2
generate de bert_cased large-11_9k 5.postcasing+editscript.1

generate en bert_uncased 2_5k 0.postcasing+postdiacritics+editscript.5
generate en bert_uncased fixed-4_4k 1.postcasing+postdiacritics+editscript.3
generate en bert_uncased fixed-7_5k 1.postcasing+postdiacritics+editscript.2
generate en bert_uncased large-32_9k 1.postcasing+postdiacritics+editscript.1

generate en bert_cased 2_5k 0.postcasing+editscript.5
generate en bert_cased 4_0k 0.postcasing+editscript.3
generate en bert_cased 6_8k 0.postcasing+editscript.2
generate en bert_cased large-32_1k 1.postcasing+editscript.1

generate en enbert_cased 2_5k 1.postcasing+editscript.5
generate en enbert_cased 4_3k 1.postcasing+editscript.3
generate en enbert_cased 4_2km 1.postcasing+editscript.3
generate en enbert_cased large-32_6k 1.postcasing+editscript.1

generate ru bert_uncased 0_5k 0.postcasing+postdiacritics+editscript.2
generate ru bert_uncased 1_6k 5.postcasing+postdiacritics+editscript.2
generate ru bert_uncased 3_3k 1.postcasing+postdiacritics+editscript.1
generate ru bert_uncased large-10_0k 9.postcasing+postdiacritics+editscript.1

generate ru bert_cased 0_5k 0.postcasing+editscript.2
generate ru bert_cased 1_5k 5.postcasing+editscript.2
generate ru bert_cased 3_1k 1.postcasing+editscript.1
generate ru bert_cased large-9_2k 9.postcasing+editscript.1

####################
# Word-level rules #
####################
generate cs bert_uncased w-3_5k 1.words+postcasing+postdiacritics+editscript.5
generate cs bert_uncased w-6_1k 1.words+postcasing+postdiacritics+editscript.3
generate cs bert_uncased w-11k 1.words+postcasing+postdiacritics+editscript.2

generate cs bert_cased w-2_3k 1.words+postcasing+editscript.5
generate cs bert_cased w-4_2k 1.words+postcasing+editscript.3
generate cs bert_cased w-8_0k 1.words+postcasing+editscript.2

generate de bert_uncased w-1_4k 1.words+postcasing+postdiacritics+editscript.4
generate de bert_uncased w-3_1k 1.words+postcasing+postdiacritics+editscript.2
generate de bert_uncased w-12_1k 1.words+postcasing+postdiacritics+editscript.1

generate de bert_cased w-1_2k 1.words+postcasing+editscript.4
generate de bert_cased w-3_6k 5.words+postcasing+editscript.2
generate de bert_cased w-10_9k 1.words+postcasing+editscript.1

generate en bert_uncased w-3_4k 1.words+postcasing+postdiacritics+editscript.4
generate en bert_uncased w-4_6k 1.words+postcasing+postdiacritics+editscript.3
generate en bert_uncased w-9_0k 1.words+postcasing+postdiacritics+editscript.2

generate en bert_cased w-3_2k 1.words+postcasing+editscript.4
generate en bert_cased w-5_1k 5.words+postcasing+editscript.3
generate en bert_cased w-8_6k 5.words+postcasing+editscript.2

generate en enbert_cased w-3_2k 1.words+postcasing+editscript.4
generate en enbert_cased w-5_1k 5.words+postcasing+editscript.3
generate en enbert_cased w-8_6k 5.words+postcasing+editscript.2

generate ru bert_uncased w-0_5k 1.words+postcasing+postdiacritics+editscript.3
generate ru bert_uncased w-2_1k 5.words+postcasing+postdiacritics+editscript.2
generate ru bert_uncased w-9_5k 5.words+postcasing+postdiacritics+editscript.1

generate ru bert_cased w-0_5k 1.words+postcasing+editscript.3
generate ru bert_cased w-1_8k 5.words+postcasing+editscript.2
generate ru bert_cased w-8_3k 5.words+postcasing+editscript.1
