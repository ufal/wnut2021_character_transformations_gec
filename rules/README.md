# Character Transformations Scripts

This directory contains scripts for aligning GEC, creating character
transformations from aligned data, encoding gold data with the transformations
and finally applying the transformations on input data.


## Alignment

To create the transformations, we first need to align input GEC data, so that
each input subword is aligned with its corresponding gold data character
sequence. This subword-to-character_sequence alignment is performed using the
Algorithm 1 from the paper, which is computed using the [aligner.py](aligner.py)
script. It has the following options:
- `--data`: input data, where each line contains an input sentence and (possible
  several) corrected gold sentence. The entries are tab-separated and
  - if `--first_correct` is given, the first entry is the gold sentence
  - otherwise, the first entry is the input sentence, followed by any number of
    gold sentences
- `--dump`: output path
- `--limit=n`: maximum number of sentences to process
- `--threads=n`: threads to use
- `--tokenizer=name`: tokenizer to use

In the paper, we used the [aligner_all.sh](aligner_all.sh) script to process all
our GEC data (which was available in `data_aligned/input` directory).


## Transformations

The [rules.py](rules.py) module implements various kinds of character
transformations. Each kind of transformation can
- create a transformation from aligned data
- apply a transformation on input data

The transformations can be composed together (but not all combinations work
correctly) and a `Rule` instance can be created from a string description
using the static `Rule.create` method. In the paper, we used the following
combinations:
- _char-at-subword_: `postcasing+editscript` for cased models and `postcasing+postdiacritics+editscript` for uncased models,
- _char-at-word_: `words+postcasing+editscript` for cased models and `words+postcasing+postdiacritics+editscript` for uncased models,
- _string-at-subword_: `replace_append`
- _string-at-word_: `words+replace_append`


### Creating Transformations

The transformations can be created from the aligned data using the
[create_rules.py](create_rules.py) script, which accepts the following
arguments:
- `--train`: path to the training alignments
- `--dump`: output path
- `--rules`: text description of the transformation kind to use
- `--occurrences=n`: prune transformations with less occurrences than the given amount

We always include two special rules:
- `KEEP` with id=0, which copies input subword to output
- `UNCORRECTABLE_ERROR` with id=1, which corresponds to an error not describable
  by the created transformations

In the paper, we used the [create_rules_all.sh](create_rules_all.sh) script to
create transformations using a grid of several hyperparameters.


### Encoding Transformations

Once the transformations are created, gold data can be encoded using the
via the [encode_rules.py](encode_rules.py) script with the following options:
- `--data`: input data to encode
- `--dump`: output path
- `--rules`: text description of the transformation kind to use
- `--rules_file`: path to the file with created transformations
- `--others`: by default, when encoding a transformation, we try creating it
  using the `Rule` instance, and look the result up in the transformation
  dictionary; if not found, then if `--others` is given, we also try to apply
  all the transformations in the dictionary and check whether one of them
  produces the correct output (this is time-consuming, but it slightly increases
  coverage)
- `--threads=n`: threads to use
- `--tokenizer=name`: tokenizer to use

In the paper, we have encoded the data with a subset of created transformations
using the [encode_rules_all.sh](encode_rules_all.sh) script.


### Applying Transformations

To apply chosen transformations on input sentences consisting of subword ids,
the [apply_rules.py](apply_rules.py) can be used. It has the following options:
- `--data`: data to process, consisting of a list of sentences, each being
  a pair of a list of subword ids and transformation ids
- `--rules_file`: path to the file with the used transformations
- `--copy_uncorrectable=0`: mark the `UNCORRECTABLE_ERROR`s transformations
  using a `[UNCORRECTABLE_ERROR]` string
- `--copy_uncorrectable=1`: when `UNCORRECTABLE_ERROR` transformation is
  encountered, copy the input subword without a change
- `--tokenizer=name`: tokenizer to use

To inspect the transformed development data, we used the
[apply_rules_alldev.sh](apply_rules_alldev.sh) script to apply the transformations
(from the `encode_rules` script) on the development set, marking the
uncorrectable errors using the `[UNCORRECTABLE_ERROR]` string. The
[apply_rules_alltest.sh](apply_rules_alltest.sh) script is analogous, only
processing the test data and copying input subwords when uncorrectable errors
are encountered (instead of using the `[UNCORRECTABLE_ERROR]` marker).


## Tokenizer

The [tokenizer.py](tokenizer.py) module is a  HuggingFace tokenizer wrapper,
which
- given a plain text sentence produces a list of subwords, which are as close
  to the original text as possible (i.e., we use explicit space at the beginning
  of subwords starting a new word)
- it can convert such subwords back to their ids
- it can process gold sequence of characters in order to correspond to the
  encoding used by the tokenizer (i.e., for RoBERTa, we apply the same transformation
  as the byte-level BPE does)
