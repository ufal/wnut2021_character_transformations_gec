#!/usr/bin/env bash

set -ex

DATA_DIR="" # TODO fill in path to folder storing .pickle.gz files with training, development and testing data and RULES.LIST file
BERT_MODEL="" # TODO fill in BERT type (e.g.bert-base-multilingual-cased)
TOKENIZER_NAME="" # TODO fill in tokenizer name (e.g. bert-base-multilingual-cased)
CLASS_WEIGHT=1 # weight for weighting non-copy instructions
FINETUNE_FROM="" # dir to finetune the model from

MAX_LENGTH=128
BATCH_SIZE=64
GRAD_ACC_STEPS=32
SAVE_STEPS=312
SEED=1
NUM_EPOCHS=20

if [[ $FINETUNE_FROM == "" ]]; then
    EXPERIMENT_NAME=$(basename $DATA_DIR)-$BERT_MODEL
else
    EXPERIMENT_NAME=$(basename $DATA_DIR)-$BERT_MODEL-ff$(basename $FINETUNE_FROM)
fi

OUTPUT_DIR="" # TODO fill in path to store model checkpoints
CACHE_DIR="" # TODO fill in path to store cache files

mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

if [[ $FINETUNE_FROM != '' ]]; then
    last_ckpt=$(ls -ldt $FINETUNE_FROM/checkpoint* | head -n 2 | tail -n 1 | rev | cut -d ' ' -f 1 | rev)
    last_ckpt=$(basename $last_ckpt)
    mkdir -p $OUTPUT_DIR/checkpoint-0 
    cp -r $FINETUNE_FROM/$last_ckpt/* $OUTPUT_DIR/checkpoint-0
    BERT_MODEL=$OUTPUT_DIR/checkpoint-0
fi

python run_training.py \
        --model_name_or_path $BERT_MODEL \
        --tokenizer_name $TOKENIZER_NAME \
        --cache_dir $CACHE_DIR \
        --max_seq_length $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --cache_dir $CACHE_DIR \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --do_train \
        --overwrite_output_dir \
        --output_dir $OUTPUT_DIR \
        --data_dir $DATA_DIR
        
