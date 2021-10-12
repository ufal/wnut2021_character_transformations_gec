#!/usr/bin/env bash

set -ex

IN_FILE="" # TODO fill in to infile (.pickle.gz)
OUT_FILE="" # TODO fill in path to outfile to store predicted instructions
DATA_DIR="" # TODO fill in path to folder storing .pickle.gz files with training, development and testing data and RULES.LIST file
MODEL="" # TODO fill in path to the trained model
MODEL_NAME="" # TODO fill in BERT type (e.g.bert-base-multilingual-cased)
MAX_LENGTH=128

CACHE_DIR="" # TODO fill in cache dir

python run_training.py \
 --model_name_or_path $MODEL \
 --tokenizer_name $MODEL_NAME \
 --output_dir $MODEL \
 --per_device_eval_batch_size 1 \
 --max_seq_length $MAX_LENGTH \
 --cache_dir $CACHE_DIR \
 --data_dir $DATA_DIR \
 --do_predict \
 --prediction_file_path $IN_FILE > $OUT_FILE
