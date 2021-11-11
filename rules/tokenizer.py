#!/usr/bin/env python3
from functools import lru_cache
from typing import List

import transformers

# Taken from transformers.tokenization_gpt2
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

@lru_cache()
def unicode_to_bytes():
    return {v: k for k, v in bytes_to_unicode().items()}


class Tokenizer:
    def apply(self, string: str):
        raise NotImplementedError()

    def process_input(self, string: str):
        raise NotImplementedError()

    def process_output(self, string: str):
        raise NotImplementedError()

    def ids_to_tokens(self, ids: List[int]):
        raise NotImplementedError()

    def tokens_to_ids(self, tokens: List[str]):
        raise NotImplementedError()

    @staticmethod
    def create(name: str):
        if name == "electra":
            return ElectraTokenizer()
        if name.startswith("bert"):
            return HFTokenizer(name, "bert-like")
        if name == "ufal/robeczech-base":
            return HFTokenizer(name, "roberta-like")

        raise RuntimeError("Unknown tokenizer {}".format(name))

class HFTokenizer:
    def __init__(self, tokenizer, kind):
        assert kind in ("bert-like", "roberta-like")

        if isinstance(tokenizer, str):
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self._tokenizer = tokenizer
        self._kind = kind
        self._token_to_id = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    def apply(self, string: str):
        tokenized = self._tokenizer.tokenize(string)
        if self._kind == "bert-like":
            return self._postprocess_bert_like(tokenized)
        elif self._kind == "roberta-like":
            return self._postprocess_roberta_like(tokenized)

    def process_input(self, string: str):
        if self._kind == "bert-like":
            return string
        elif self._kind == "roberta-like":
            mapping = bytes_to_unicode()
            return "".join(mapping[b] for b in string.encode("utf-8")).replace('Ġ', ' ')

    def process_output(self, string: str):
        if self._kind == "bert-like":
            return string
        elif self._kind == "roberta-like":
            mapping = unicode_to_bytes()
            return bytes(mapping[b] for b in string.replace(' ', 'Ġ')).decode("utf-8", errors="ignore")

    def ids_to_tokens(self, ids: List[int]):
        tokens = []
        for token_id in ids:
            tokens.append(self._id_to_token[token_id])
        if self._kind == "bert-like":
            return self._postprocess_bert_like(tokens)
        elif self._kind == "roberta-like":
            return self._postprocess_roberta_like(tokens)

    def tokens_to_ids(self, tokens: List[str]):
        ids = []
        if tokens:
            ids.append(self._token_to_id[tokens[0]])
            for i in range(1, len(tokens)):
                if self._kind == "bert-like":
                    if tokens[i].startswith(' '):
                        ids.append(self._token_to_id[tokens[i][1:]])
                    else:
                        ids.append(self._token_to_id["##" + tokens[i]])
                elif self._kind == "roberta-like":
                    if tokens[i].startswith(' '):
                        ids.append(self._token_to_id['Ġ' + tokens[i][1:]])
                    else:
                        ids.append(self._token_to_id[tokens[i]])
        return ids

    def _postprocess_bert_like(self, tokenized:list):
        spaces_correct = []
        for token_ind, token in enumerate(tokenized):
            if token_ind == 0:
                spaces_correct.append(token)
                continue

            if token.startswith('##'):
                spaces_correct.append(token[2:])
            else:
                spaces_correct.append(' ' + token)
        return spaces_correct

    def _postprocess_roberta_like(self, tokenized:list):
        spaces_correct = []
        for token_ind, token in enumerate(tokenized):
            if token_ind == 0:
                spaces_correct.append(token)
                continue

            if token.startswith('Ġ'):
                spaces_correct.append(' ' + token[1:])
            else:
                spaces_correct.append(token)
        return spaces_correct


class ElectraTokenizer(HFTokenizer):
    def __init__(self):
        super().__init__(transformers.ElectraTokenizerFast(
            '/home/straka/students/naplava/bert_gec/tokenizer/vocab.txt', strip_accents = False, do_lower_case = False), "bert-like")
