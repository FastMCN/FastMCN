import re
import sys
import pickle
import json
import numpy as np
from utils import *
from pprint import pprint
from pathlib import Path
from functools import reduce
from operator import itemgetter
from itertools import groupby, chain, count
from collections import defaultdict
from pprint import pprint
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ZERO_PAD = True

file_name = sys.argv[1] if len(
    sys.argv) > 1 else "../proc_data/cdr_ctd_data_v3.csv"

test_dev_file = sys.argv[2] if len(
    sys.argv) > 2 else "../proc_data/cdr_data_v3.csv"

proc_path = sys.argv[3] if len(
    sys.argv) > 3 else "../proc_data/tmp/"

vocab_size = sys.argv[4] if len(
    sys.argv) > 4 else 32768

proc_path = f"{proc_path}_{vocab_size}"

char_map_path = sys.argv[4] if len(sys.argv) > 4 else None

path = Path(proc_path)

if not path.exists():
    path.mkdir(parents=True)


def unique(obj, prefix="", verbose=True):
    s = set(chain(*obj))
    if verbose:
        print(f"{prefix}{len(s)}")
    return s


def split_to_words(x: str):
    return [w for w in __split_to_words(x) if w not in stop_words]


class GenData():
    def __init__(self, name_bpe_words, word_map, bpe_vocab):
        self.name_bpe_words = name_bpe_words
        self.word_map = word_map
        self.bpe_vocab = bpe_vocab
        self.append_funcs = {"train": self.train_append,
                             "dev": self.test_dev_append,
                             "test": self.test_dev_append}

    def gen_result(self, name, concept):
        bpe_words = self.name_bpe_words[name]
        result = {"name": name,
                  "word": bpe_words,
                  "n_word": len(bpe_words),
                  "char_code": [self.word_map[w] for w in bpe_words],
                  "word_code": [self.bpe_vocab[w] for w in bpe_words],
                  "concept": concept}
        return result

    def append(self, dd, dt):
        dd = self.append_funcs[dt['set']](dd, dt)
        return dd

    def train_append(self, dd, dt):
        concept = dt['concept']
        mention = dt['mention']
        if mention in self.name_bpe_words and ("|" not in concept):
            dd[dt['set']][concept].append(self.gen_result(mention, concept))
        if len(dd[dt['set']][concept]) > 1:
            return dd
        if concept != mention and ("|" not in concept):
            dd[dt['set']][concept].append(self.gen_result(concept, concept))
        return dd

    def test_dev_append(self, dd, dt):
        concept = dt['concept']
        mention = dt['mention']
        dd[dt['set']]['data'].append(self.gen_result(mention, concept))
        return dd

    def __call__(self, data):
        dd = defaultdict(lambda: defaultdict(list))
        dd = reduce(self.append, data, dd)
        return dd


def dump_by_name(proc_path: str, obj_name: str):
    with open(f"{proc_path}/{obj_name}.pkl", "wb") as f:
        pickle.dump(globals()[obj_name], f)


__split_to_words = RegSplit(pattern="[a-zA-Z']+|[0-9]+|[^a-zA-Z0-9]",
                            exclude=[" ", ",", "-", "/", '\n', '&', ".",
                                     '(', ')', '+', ':', ';', '[', ']'])

stop_words = {"a", "an", "and", "are", "as", "at", "be", "but", "by",
              "for", "if", "in", "into", "is", "it", "of", "on", "or",
              "such", "that", "the", "their", "then", "there",
              "these", "they", "this", "to", "was", "will", "with"}

data = read_csv(file_name)
data = [i for i in data if i['set'] == "train"]

test_dev = read_csv(test_dev_file)
test_dev = [i for i in test_dev if i['set'] != "train"]

data += test_dev

mentions = [i['mention'].lower() for i in data]
print(f"mentions: {len(mentions)}")
print(f"Unique mentions: {len(set(mentions))}\n")

concepts = set([c.lower() for i in data for c in i['concept'].split("|")])
print(f"Unique concepts: {len(concepts)}\n")

# name = mention + concept
names = unique([mentions + list(concepts)], verbose=False) - stop_words

# names = unique([mentions + list(concepts)], verbose=False)
print(f"Unique names: {len(names)}\n")

name_words = {n: " ".join(split_to_words(n))
              for n in names}

with open(f"{proc_path}/names.txt", "w") as f:
    f.write("\n".join(list(name_words.values())))
    # f.write("\n".join(words))

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    # NFKC(),
    Lowercase()
])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=int(vocab_size), show_progress=True)
tokenizer.train(trainer, [f"{proc_path}/names.txt"])

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

tokenizer.model.save(proc_path)

tokenizer.model = BPE.from_file(
    f'{proc_path}/vocab.json', f'{proc_path}/merges.txt')

with open(f"{proc_path}/vocab.json", "r") as f:
    bpe_vocab = json.load(f)

bpe_vocab_idx = {v: k for k, v in bpe_vocab.items()}

char_map = {k: v + 1
            for k, v in bpe_vocab.items() if len(k) == 1}
print(f"Char map size: {len(char_map)}\n")

MAX_LEN_OF_WORD = max([len(w) for w in bpe_vocab])
print(f"Max length of word: {MAX_LEN_OF_WORD}\n")

if ZERO_PAD:
    word_map = {k: [char_map[c] for c in k] + [0] * (MAX_LEN_OF_WORD - len(k))
                for k in bpe_vocab}
else:
    word_map = {k: [char_map[c] for c in k]
                for k in bpe_vocab}

name_bpe_words = {n: tokenizer.encode(w).tokens for n, w in name_words.items()}
MAX_LEN_OF_MENTION = max([len(v) for v in name_bpe_words.values()])
print(f"Max length of names: {MAX_LEN_OF_MENTION}\n")

gen_data = GenData(name_bpe_words, word_map, bpe_vocab)
data_sets = gen_data(data)
# train_data, test_data, dev_data = [
#     dict(dt) for dt in list(gen_data(data).values())]
train_data, test_data, dev_data = [
    data_sets[k] for k in ['train', "test", "dev"]]

global_var = {"MAX_LEN_OF_MENTION": MAX_LEN_OF_MENTION,
              "MAX_LEN_OF_WORD": MAX_LEN_OF_WORD}

dump_vars = ["train_data", "dev_data", "test_data", "char_map",
             "word_map", "global_var", "bpe_vocab", "bpe_vocab_idx"]

print(f"Dumping data at {proc_path}:")
for i in dump_vars:
    print(i)
    dump_by_name(proc_path, i)

# Local Variables:
# elpy-shell-starting-directory: current-directory
# pyvenv-activate: "/home/chongliang/miniconda3/envs/disease_normalization/"
# End:
