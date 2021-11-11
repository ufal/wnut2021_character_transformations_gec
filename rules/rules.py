from typing import List
import difflib
import functools
import json
import unicodedata

import Levenshtein

ASSERT = False

class Rule:
    def __init__(self, orig:List[str], cor:List[str], index:int):
        pass

    def apply(self, text:List[str], index:int) -> str:
        raise NotImplementedError()

    @staticmethod
    def is_applicable(orig:List[str], cor:List[str], index:int) -> bool:
        return True

    def serialize_fields(self):
        raise NotImplementedError()

    def serialize(self):
        return [
            self.__class__.__name__,
            {field: value for field in self.serialize_fields() for value in [getattr(self, field)] if not isinstance(value, Rule)},
            {field: value.serialize() for field in self.serialize_fields() for value in [getattr(self, field)] if isinstance(value, Rule)},
        ]

    def to_str(self):
        return json.dumps(self.serialize(), ensure_ascii=False, sort_keys=True, indent=None)

    @staticmethod
    def from_str(text:str):
        instance_name, primitives, rules = json.loads(text)
        instance_type = globals()[instance_name]
        instance = instance_type.__new__(instance_type)
        for key, value in primitives.items():
            setattr(instance, key, value)
        for key, value in rules.items():
            setattr(instance, key, Rule.from_str(json.dumps(value)))
        return instance

    @staticmethod
    def create(name:str):
        if name == "replace":
            return ReplaceWholeRule
        if name == "replace_append":
            return ReplaceAppendWholeRule
        if name == "editscript_abs":
            return functools.partial(CharEditScriptRule, negative_indices=False)
        if name == "editscript":
            return functools.partial(CharEditScriptRule, negative_indices=True)
        for rule, cls in [("casing", ExtraCasingRule), ("diacritics", ExtraDiacriticsRule)]:
            for rule_abs, negative_indices in [("", True), ("_abs", False)]:
                for rule_pre, pre in [("pre", True), ("post", False)]:
                    rule_name = rule_pre + rule + rule_abs + "+"
                    if name.startswith(rule_name):
                        return functools.partial(cls, base=Rule.create(name[len(rule_name):]), pre=pre, negative_indices=negative_indices)
        if name.startswith("split+"):
            return functools.partial(SplitRule, base=Rule.create(name[len("split+"):]))
        for window in range(1, 9+1):
            for rationame, ratio in [("", 2), ("edit80", 0.79)]:
                rule_name = "move{}{}+".format(window, rationame)
                if name.startswith(rule_name):
                    return functools.partial(MoveRule, base=Rule.create(name[len(rule_name):]), window=window, ratio=ratio)
        if name.startswith("words+"):
            return functools.partial(WordsRule, base=Rule.create(name[len("words+"):]))
        if name.startswith("rev+"):
            return functools.partial(RevRule, base=Rule.create(name[len("rev+"):]))

        raise ValueError("Unknown rule name '{}'".format(name))

class ReplaceWholeRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int):
        orig, cor = orig[index], cor[index]

        self._replace = None if orig == cor else cor

    def apply(self, text:List[str], index:int) -> str:
        text = text[index]

        return text if self._replace is None else self._replace

    def serialize_fields(self):
        return ["_replace"]


class ReplaceAppendWholeRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int):
        orig, cor = orig[index], cor[index]

        self._replace, self._append_before, self._append_after = None, None, None
        if orig != cor:
            if orig and cor.startswith(orig):
                self._append_after = cor[len(orig):]
            elif orig and cor.endswith(orig):
                self._append_before = cor[:-len(orig)]
            else:
                self._replace = cor

    def apply(self, text:List[str], index:int) -> str:
        text = text[index]

        if self._append_after is not None:
            text = text + self._append_after
        elif self._append_before is not  None:
            text = self._append_before + text
        elif self._replace is not None:
            text = self._replace

        return text

    def serialize_fields(self):
        return ["_replace", "_append_before", "_append_after"]


class CharEditScriptRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int, negative_indices:bool):
        orig, cor = orig[index], cor[index]

        self.edits = []
        if orig != cor and Levenshtein.ratio(orig, cor) < 0.3:
            self.edits.append(['replace_all', cor])
        elif orig != cor:
            sm = difflib.SequenceMatcher(None, orig, cor, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                i1_orig = i1
                i2_orig = i2
                if negative_indices:
                    if i1 >= (len(orig) + 1) // 2:
                        i1 = i1 - len(orig) - 1
                        i2 = i2 - len(orig) - 1
                if tag != 'equal':
                    if tag == 'delete':
                        if i1_orig == 0 and i2_orig == len(orig):  # special case - delete whole word
                            self.edits.append(['delete_all'])
                        else:
                            self.edits.append([tag, i1, i2])
                    elif tag == 'insert':
                        self.edits.append([tag, i1, cor[j1:j2]])
                    elif tag == 'replace':
                        if i1_orig == 0 and i2_orig == len(orig):
                            self.edits.append(['replace_all', cor[j1:j2]])
                        else:
                            self.edits.append([tag, i1, i2, cor[j1:j2]])
                    else:
                        raise ValueError(f'Unknown value of tag {tag}')
        if ASSERT:
            assert self.apply([orig], 0) == cor

    def apply(self, text:List[str], index:int) -> str:
        text = text[index]

        text_splitted = list(text) + ['']  # last because of insert to the end of text instruction
        for edit, *args in self.edits:
            if edit == 'delete_all':
                return ''
            elif edit == 'replace_all':
                return args[0]
            elif edit == 'delete':
                if args[0] < -len(text_splitted): return None
                if args[1] > len(text_splitted): return None
                for i in range(args[0], args[1]):
                    text_splitted[i] = ''
            elif edit == 'insert':
                if args[0] >= len(text_splitted) or args[0] < -len(text_splitted): return None
                text_splitted[args[0]] = args[1] + text_splitted[args[0]]
            elif edit == 'replace':
                if args[0] < -len(text_splitted): return None
                if args[1] > len(text_splitted): return None
                for i in range(args[0], args[1]):
                    text_splitted[i] = ''
                text_splitted[args[0]] = args[2]
        return ''.join(text_splitted)

    def serialize_fields(self):
        return ["edits"]


class ExtraCasingRule(Rule):
    @staticmethod
    def safer_lower(text):
        return text.replace("İ", "i").lower()

    def __init__(self, orig:List[str], cor:List[str], index:int, pre:bool, negative_indices:bool, base:Rule):
        self.pre = pre

        orig, cor = orig[index], cor[index]

        self.base = base([self.safer_lower(orig)], [self.safer_lower(cor)], 0)
        if pre:
            orig_in = self.safer_lower(orig)
            cor_out = self.safer_lower(cor)
        else:
            orig_in = orig
            cor_out = self.base.apply([orig], 0)

        self.edits = []
        is_lower = cor.islower() if len(cor) and not pre else True
        for i in range(len(cor)):
            if pre:
                if is_lower and cor[i].isupper():
                    self.edits.append(("upper", i - len(cor) if i >= (len(cor) + 1) // 2 else i))
                    is_lower = False
                elif not is_lower and cor[i].islower():
                    self.edits.append(("lower", i - len(cor) if i >= (len(cor) + 1) // 2 else i))
                    is_lower = True
            else:
                if cor_out[i].isupper() != cor[i].isupper():
                    self.edits.append(("upper" if cor[i].isupper() else "lower", i - len(cor) if i >= (len(cor) + 1) // 2 else i))

        if ASSERT:
            assert self.apply([orig_in], 0) == cor, (self.apply([orig_in], 0), cor, self.edits)

    def apply(self, text:List[str], index:int) -> str:
        text = text[index]

        if self.pre:
            text = self.safer_lower(text)
        text = self.base.apply([text], 0)
        if text is None: return None

        for edit, index in self.edits:
            if index >= len(text) or index < -len(text): return None

            if self.pre:
                text = text[:index] + (str.upper if edit == "upper" else str.lower)(text[index:])
            else:
                text = text[:index] + (str.upper if edit == "upper" else str.lower)(text[index]) + (text[index+1:] if index != -1 else "")
        return text

    def serialize_fields(self):
        return ["edits", "pre", "base"]


class ExtraDiacriticsRule(Rule):
    diacritics_remove, diacritics_name, diacritics_add = {}, {}, {}
    @staticmethod
    def init_tables():
        for i in range(0x110000):
            c = chr(i)
            name = unicodedata.name(c, "")
            with_index = name.find(" WITH ")
            if with_index >= 0:
                try:
                    c_no_diacritics = unicodedata.lookup(name[:with_index])
                    diacritics_name = name[with_index + 6:]

                    ExtraDiacriticsRule.diacritics_remove[c] = c_no_diacritics
                    ExtraDiacriticsRule.diacritics_name[c] = diacritics_name

                    if c_no_diacritics not in ExtraDiacriticsRule.diacritics_add:
                        ExtraDiacriticsRule.diacritics_add[c_no_diacritics] = {}
                    ExtraDiacriticsRule.diacritics_add[c_no_diacritics][diacritics_name] = c

                except KeyError:
                    pass

    @staticmethod
    def remove_diacritics(text):
        return "".join(ExtraDiacriticsRule.diacritics_remove.get(c, c) for c in text)

    def __init__(self, orig:List[str], cor:List[str], index:int, pre:bool, negative_indices:bool, base:Rule):
        self.pre = pre

        orig, cor = orig[index], cor[index]

        orig_no_diacritics = "".join(self.diacritics_remove.get(c, c) for c in orig)
        cor_no_diacritics = "".join(self.diacritics_remove.get(c, c) for c in cor)

        self.base = base([orig_no_diacritics], [cor_no_diacritics], 0)
        if pre:
            orig_in = orig_no_diacritics
            cor_out = cor_no_diacritics
        else:
            orig_in = orig
            cor_out = self.base.apply([orig], 0)

        self.edits = []
        for i in range(len(cor)):
            if cor[i] == cor_out[i]:
                continue

            diacritics_name = self.diacritics_name[cor[i]] if cor[i] in self.diacritics_name else ""
            self.edits.append((diacritics_name, i - len(cor) if i >= (len(cor) + 1) // 2 else i))
        if ASSERT:
            assert self.apply([orig_in], 0) == cor, (self.apply([orig_in], 0), orig_in, cor, self.edits)

    def apply(self, text:List[str], index:int) -> str:
        text = text[index]

        if self.pre:
            text = "".join(self.diacritics_remove.get(c, c) for c in text)
        text = self.base.apply([text], 0)
        if text is None: return None
        text = list(text)

        for diacritics, index in self.edits:
            if index >= len(text) or index < -len(text): return None

            replacement = self.diacritics_remove.get(text[index], text[index])

            if diacritics:
                replacement = self.diacritics_add.get(replacement, None)
                if replacement is None: return None

                replacement = replacement.get(diacritics, None)
                if replacement is None: return None

            text[index] = replacement
        return "".join(text)

    def serialize_fields(self):
        return ["edits", "pre", "base"]
ExtraDiacriticsRule.init_tables()

class SplitRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int, base:Rule):
        self.edits = []

        word, words = cor[index], []
        while word:
            space = word.find(' ', 1)
            if space < 0: space = len(word)
            assert word[:space], (cor[index], word, space)
            words.append(word[:space])
            word = word[space:]

        for word in words:
            setattr(self, "base{}".format(len(self.edits)), base([orig[index]], [word], 0))
            self.edits.append((0, 0, len(self.edits)))

        if ASSERT:
            assert self.apply(orig, index) == cor[index], (self.apply(orig, index), cor[index], words, orig[index], self.edits, self.base0.serialize())

    def apply(self, text:List[str], index:int) -> str:
        result = []
        for start, end, base in self.edits:
            if index + start < 0: return None
            if index + end >= len(text): return None
            result.append(getattr(self, "base{}".format(base)).apply(["".join(text[index + start:index + end + 1])], 0))
        return "".join(result) if all(x is not None for x in result) else None

    def serialize_fields(self):
        return ["edits"] + ["base{}".format(base) for _, _, base in self.edits]

class MoveRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int, base:Rule, window:int, ratio:float):
        word, words = cor[index], []
        while word:
            space = word.find(' ', 1)
            if space < 0: space = len(word)
            assert word[:space], (cor[index], word, space)
            words.append(word[:space])
            word = word[space:]

        # Prefixes
        prefixes = []
        while words:
            start, end = self._best_range(words[0], orig, index, window, ratio)
            if start is None: break
            prefixes.append((start, end, base(["".join(orig[index + start:index + end + 1])], [words[0]], 0)))
            words = words[1:]

        # Suffixes
        suffixes = []
        while words:
            start, end = self._best_range(words[-1], orig, index, window, ratio)
            if start is None: break
            suffixes.append((start, end, base(["".join(orig[index + start:index + end + 1])], [words[-1]], 0)))
            words = words[:-1]
        suffixes.reverse()

        if words:
          self.edits = [(0, 0, base(orig, cor, index))]
        else:
          self.edits = prefixes + suffixes

#         # Rest
#         if words:
#             prefixes.append((0, 0, base([orig[index]], ["".join(words)], 0)))
#         self.edits = prefixes + suffixes

        # Create rules
        for i, (start, end, rule) in enumerate(self.edits):
            setattr(self, "base{}".format(i), rule)
            self.edits[i] = (start, end, i)

        if ASSERT:
            assert self.apply(orig, index) == cor[index], (self.apply(orig, index), cor[index], words, orig[index], self.edits, self.base0.serialize())

    def _best_range(self, word:str, orig:List[str], index:int, window:int, ratio:float):
        best_start, best_end = None, None
        for start in range(-window, window + 1):
            if index + start < 0: continue
            for end in range(start, window + 1):
                if index + end >= len(orig): break
                orig_joined = "".join(orig[index + start:index + end + 1])
                if len(orig_joined) > len(word): break
                if (ratio >= 1 and orig_joined == word) or \
                        (ratio < 1 and Levenshtein.ratio(orig_joined, word) >= ratio):
                    if best_start is None or abs((end - start) / 2) < abs((best_end - best_start) / 2):
                        best_start, best_end = start, end
        return best_start, best_end

    def apply(self, text:List[str], index:int) -> str:
        result = []
        for start, end, base in self.edits:
            if index + start < 0: return None
            if index + end >= len(text): return None
            result.append(getattr(self, "base{}".format(base)).apply(["".join(text[index + start:index + end + 1])], 0))
        return "".join(result) if all(x is not None for x in result) else None

    def serialize_fields(self):
        return ["edits"] + ["base{}".format(base) for _, _, base in self.edits]

class WordsRule(Rule):
    _cache_init = None

    def __init__(self, orig:List[str], cor:List[str], index:int, base:Rule):
        if WordsRule._cache_init and WordsRule._cache_init[0] == (orig, cor):
            mapped_orig, mapped_cor, mapped_indices = WordsRule._cache_init[1]
        else:
            mapped_orig, mapped_cor, mapped_indices = [], [], []
            for i, subword in enumerate(orig):
                if i == 0 or subword.startswith(" "):
                    mapped_indices.append(len(mapped_orig))
                    mapped_orig.append((" " if i == 0 else "") + subword)
                    mapped_cor.append((" " if i == 0 else "") + cor[i])
                else:
                    mapped_indices.append(-1)
                    mapped_orig[-1] += subword
                    mapped_cor[-1] += cor[i]
            WordsRule._cache_init = ((orig, cor), (mapped_orig, mapped_cor, mapped_indices))

        if mapped_indices[index] >= 0:
            self.base = base(mapped_orig, mapped_cor, mapped_indices[index])
        else:
            self.base = base(["REMOVE_ALL"], [""], 0)

    _cache_apply = None

    def apply(self, text:List[str], index:int) -> str:
        if WordsRule._cache_apply and WordsRule._cache_apply[0] == text:
            mapped_text, mapped_indices = WordsRule._cache_apply[1]
        else:
            mapped_text, mapped_indices = [], []
            for i, subword in enumerate(text):
                if i == 0 or subword.startswith(" "):
                    mapped_indices.append(len(mapped_text))
                    mapped_text.append((" " if i == 0 else "") + subword)
                else:
                    mapped_indices.append(-1)
                    mapped_text[-1] += subword
            WordsRule._cache_apply = (text, (mapped_text, mapped_indices))

        if mapped_indices[index] >= 0:
            return self.base.apply(mapped_text, mapped_indices[index])
        else:
            return ""

    def serialize_fields(self):
        return ["base"]

class RevRule(Rule):
    def __init__(self, orig:List[str], cor:List[str], index:int, base:Rule):
        self.base = base([orig[index][::-1]], [cor[index][::-1]], 0)

    def apply(self, text:List[str], index:int) -> str:
        result = self.base.apply([text[index][::-1]], 0)
        if result is not None:
            result = result[::-1]
        return result

    def serialize_fields(self):
        return ["base"]
