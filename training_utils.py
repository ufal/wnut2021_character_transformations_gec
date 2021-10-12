# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Diacritization fine-tuning. """

import glob
import gzip
import logging
import os
import pickle
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
from dataclasses import dataclass
from filelock import FileLock
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

np.random.seed(42)

logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    labels: Optional[List[int]] = None


@dataclass
class PredictInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GECDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            cache_dir: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,

    ):
        '''

        :param lang: either concrete language (e.g. cs) or 'all' that means include all languages
        :param tokenizer:
        :param labels:
        :param model_type:
        :param max_seq_length:
        :param overwrite_cache:
        :param mode:
        '''

        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        # Load data features from cache or dataset file
        cached_examples_file = os.path.join(
            cache_dir, "cached_{}_{}_{}".format(mode.value, str(max_seq_length), data_dir.replace('/', '_'))
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_examples_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_examples_file) and not overwrite_cache:
                logger.info(f"Loading examples from cached file {cached_examples_file}")
                with open(cached_examples_file, 'rb') as f:
                    self.examples = pickle.load(f)
            else:
                logger.info(f"Creating examples from dataset file at {data_dir}")

                if isinstance(mode, Split):
                    mode = mode.value

                logger.info(f"Processing {mode} data")

                self.examples = read_examples_from_disk(data_dir, mode)

                logger.info(f"Saving features into cached file {cached_examples_file}")

                with open(cached_examples_file, 'wb') as f:
                    pickle.dump(self.examples, f)

                    # torch.save(self.examples, cached_examples_file)
            self.random_examples_permutation = np.arange(len(self.examples))
            np.random.shuffle(self.random_examples_permutation)

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> InputFeatures:
        # start_time = time.time()
        while True:
            try:
                example = self.examples[self.random_examples_permutation[i]]

                features = convert_example_to_features(
                    example,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_id=self.cls_token_id,
                    sep_token_id=self.sep_token_id,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )

                break

                # logging.info("time per item: {}".format(time.time() - start_time))
            except Exception as e:
                raise e
                # print("Trying another example i+1")
                # i += 1

        return features


class FileIteratorDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            file: str,
            tokenizer,
            model_type: str
    ):
        '''

        :param lang: either concrete language (e.g. cs) or 'all' that means include all languages
        :param data_dir:
        :param tokenizer:
        :param labels:
        :param model_type:
        :param max_seq_length:
        :param overwrite_cache:
        :param mode:
        '''

        self.tokenizer = tokenizer
        self.model_type = model_type

        with gzip.open(file, "rb") as data_file:
            data = pickle.load(data_file)
            self.examples = data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # start_time = time.time()
        example = self.examples[i]

        input_subwords = example[:, 0].tolist()

        ## START PARAMS
        cls_token_at_end = bool(self.model_type in ["xlnet"])
        cls_token = self.tokenizer.cls_token
        cls_token_id = self.tokenizer.cls_token_id
        cls_token_segment_id = 2 if self.model_type in ["xlnet"] else 0
        sep_token = self.tokenizer.sep_token
        sep_token_id = self.tokenizer.sep_token_id
        sep_token_extra = False
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left = bool(self.tokenizer.padding_side == "left")
        pad_token = self.tokenizer.pad_token_id
        pad_token_segment_id = self.tokenizer.pad_token_type_id
        pad_token_label_id = self.pad_token_label_id
        sequence_a_segment_id = 0
        mask_padding_with_zero = True
        ## END PARAMS

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        input_subwords += [sep_token_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            input_subwords += [sep_token_id]
        segment_ids = [sequence_a_segment_id] * len(input_subwords)

        if cls_token_at_end:
            input_subwords += [cls_token_id]
            segment_ids += [cls_token_segment_id]
        else:
            input_subwords = [cls_token_id] + input_subwords
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = input_subwords
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if "token_type_ids" not in self.tokenizer.model_input_names:
            # segment_ids = None
            return {"input_ids": torch.Tensor([input_ids]).long(),
                    "attention_mask": torch.Tensor([input_mask]).long()}

        # logging.info("time per item: {}".format(time.time() - start_time))
        return {"input_ids": torch.Tensor([input_ids]).long(),
                "attention_mask": torch.Tensor([input_mask]).long(),
                "token_type_ids": torch.Tensor([segment_ids]).long()}


def read_examples_from_disk(
        data_dir: str,
        mode: Split):

    examples = []
    files = []
    if mode == Split.test or mode == 'test':
        for f in glob.glob(data_dir + '/test*.pickle.gz'):
            files.append(f)
    elif mode == Split.dev or mode == 'dev':
        for f in glob.glob(data_dir + '/test*.pickle.gz'):
            files.append(f)
    elif mode == Split.train or mode == 'train':
        for f in glob.glob(data_dir + '/train*.pickle.gz'):
            files.append(f)
    else:
        raise ValueError(f'Unknown mode {mode}')

    for f in files:
        with gzip.open(f, "rb") as data_file:
            data = pickle.load(data_file)
            for data_example in data:
                if data_example.shape[0] != 0:
                    examples.append(data_example)

    return examples


def convert_example_to_features(
        example: Tuple[str, str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_id: int,
        sep_token_id: int,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
) -> InputFeatures:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    input_subwords = example[:, 0].tolist()

    target_subwords_list = []
    for i in range(1, example.shape[1]):
        target_subwords_list.append(example[:, i].tolist())

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(input_subwords) > max_seq_length - special_tokens_count:
        input_subwords = input_subwords[: (max_seq_length - special_tokens_count)]
        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = target_subwords_list[i]
            target_subwords_list_new.append(target_subwords[: (max_seq_length - special_tokens_count)])
        target_subwords_list = target_subwords_list_new

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.


    input_subwords += [sep_token_id]

    target_subwords_list_new = []
    for i in range(len(target_subwords_list)):
        target_subwords = target_subwords_list[i] + [pad_token_label_id]
        target_subwords_list_new.append(target_subwords)
    target_subwords_list = target_subwords_list_new

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        input_subwords += [sep_token_id]

        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = target_subwords_list[i] + [pad_token_label_id]
            target_subwords_list_new.append(target_subwords)
        target_subwords_list = target_subwords_list_new

    segment_ids = [sequence_a_segment_id] * len(input_subwords)

    if cls_token_at_end:
        input_subwords += [cls_token_id]

        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = target_subwords_list[i] + [pad_token_label_id]
            target_subwords_list_new.append(target_subwords)
        target_subwords_list = target_subwords_list_new

        segment_ids += [cls_token_segment_id]
    else:
        input_subwords = [cls_token_id] + input_subwords

        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = [pad_token_label_id] + target_subwords_list[i]
            target_subwords_list_new.append(target_subwords)
        target_subwords_list = target_subwords_list_new

        segment_ids = [cls_token_segment_id] + segment_ids

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_subwords)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_subwords)
    if pad_on_left:
        input_subwords = ([pad_token] * padding_length) + input_subwords
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = target_subwords_list[i]
            target_subwords = ([pad_token_label_id] * padding_length) + target_subwords
            target_subwords_list_new.append(target_subwords)
        target_subwords_list = target_subwords_list_new
    else:
        input_subwords += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length

        target_subwords_list_new = []
        for i in range(len(target_subwords_list)):
            target_subwords = target_subwords_list[i]
            target_subwords += [pad_token_label_id] * padding_length
            target_subwords_list_new.append(target_subwords)
        target_subwords_list = target_subwords_list_new

    assert len(input_subwords) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(target_subwords_list[0]) == max_seq_length

    if len(target_subwords_list) == 1:
        target_subwords = target_subwords_list[0]
    else:
        target_subwords = target_subwords_list

    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

    res = InputFeatures(input_ids=input_subwords, attention_mask=input_mask, token_type_ids=segment_ids,
                        labels=target_subwords)

    return res
