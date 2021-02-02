# import


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*import][import:1]]
import numpy as np
from collections import defaultdict
from itertools import count, chain
from torch.utils.data import Dataset
# import:1 ends here

# Data


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*Data][Data:1]]
class Data(Dataset):
    def __init__(self, data, batch_size, max_size_for_each_concept):
        self.data = data
        self.len = len(data)
        self.idx = np.array(list(range(self.len)))
        self.batch_size = batch_size
        self.max_size_for_each_concept = max_size_for_each_concept
        self.gen_data = self.gen_data_fn()
        self.gen_idx = self.gen_idx_fn()

    def gen_idx_fn(self):
        while True:
            np.random.shuffle(self.idx)
            for i in self.idx:
                yield i

    def gen_dt_fn(self, dt):
        n_names = len(dt['n_words'])
        if n_names <= self.max_size_for_each_concept:
            return [dt['data'], dt['n_words'], [n_names]]
        else:
            idx = np.random.randint(0, high=n_names,
                                    size=self.max_size_for_each_concept)
            data = [dt['data'][i] for i in idx]
            n_words = [dt['n_words'][i] for i in idx]
            return [data, n_words, [self.max_size_for_each_concept]]

    def gen_data_fn(self):
        data, n_words, n_names = [[], [], []]
        # idxs = []
        # idx = self.gen_idx.__next__()
        # data, n_words, n_names = self.gen_dt_fn(self.data[idx])
        # idxs.append(idx)
        total_names = 0
        sample_counter = 0
        while True:
            # new_idx = self.gen_idx.__next__()
            # new_data, new_n_words, new_n_names = self.gen_dt_fn(
            #     self.data[new_idx])
            new_data, new_n_words, new_n_names = self.gen_dt_fn(
                self.data[self.gen_idx.__next__()])
            total_names += sum(new_n_names)
            if total_names <= self.batch_size and sample_counter <= self.len:
                data += new_data
                n_words += new_n_words
                n_names += new_n_names
                # idxs.append(new_idx)
                sample_counter += 1
            else:
                data = np.concatenate(data)
                n_words = np.array(n_words)
                n_names = np.array(n_names)
                yield data, n_words, n_names
                # idxs = np.array(idxs)
                # yield data, n_words, n_names, idxs
                data, n_words, n_names = [[], [], []]
                data += new_data
                n_words += new_n_words
                n_names += new_n_names
                total_names = sum(n_names)
                # idxs = [new_idx]
                sample_counter = 1

    def __getitem__(self, index):
        data, n_words, n_names = self.gen_data.__next__()
        return data, n_words, n_names

    def __len__(self):
        return len(self.data)
# Data:1 ends here

# DataUniqueWord


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*DataUniqueWord][DataUniqueWord:1]]
class DataUniqueWord(Dataset):
    """unique and encode words.
dataset: ncbi_ctd3"""
    def __init__(self, data, batch_size, max_size_for_each_concept):
        self.data = data
        self.len = len(data)
        self.idx = np.array(list(range(self.len)))
        self.batch_size = batch_size
        self.max_size_for_each_concept = max_size_for_each_concept
        self.gen_data = self.gen_data_fn()
        self.gen_idx = self.gen_idx_fn()

    def gen_idx_fn(self):
        while True:
            np.random.shuffle(self.idx)
            for i in self.idx:
                yield i

    def gen_dt_fn(self, dt):
        n_names = len(dt['n_words'])
        if n_names <= self.max_size_for_each_concept:
            return [dt['data'], dt['n_words'], [n_names]]
        else:
            idx = np.random.randint(0, high=n_names,
                                    size=self.max_size_for_each_concept)
            data = [dt['data'][i] for i in idx]
            n_words = [dt['n_words'][i] for i in idx]
            return [data, n_words, [self.max_size_for_each_concept]]

    def gen_data_fn(self):
        data, n_words, n_names = [[], [], []]
        # idxs = []
        # idx = self.gen_idx.__next__()
        # data, n_words, n_names = self.gen_dt_fn(self.data[idx])
        # idxs.append(idx)
        total_names = 0
        sample_counter = 0
        while True:
            # new_idx = self.gen_idx.__next__()
            # new_data, new_n_words, new_n_names = self.gen_dt_fn(
            #     self.data[new_idx])
            new_data, new_n_words, new_n_names = self.gen_dt_fn(
                self.data[self.gen_idx.__next__()])
            total_names += sum(new_n_names)
            if total_names <= self.batch_size and sample_counter <= self.len:
                data += new_data
                n_words += new_n_words
                n_names += new_n_names
                # idxs.append(new_idx)
                sample_counter += 1
            else:
                data = np.concatenate(data)
                n_words = np.array(n_words)
                n_names = np.array(n_names)
                yield data, n_words, n_names
                # idxs = np.array(idxs)
                # yield data, n_words, n_names, idxs
                data, n_words, n_names = [[], [], []]
                data += new_data
                n_words += new_n_words
                n_names += new_n_names
                total_names = sum(n_names)
                # idxs = [new_idx]
                sample_counter = 1

    def __getitem__(self, index):
        data, n_words, n_names = self.gen_data.__next__()
        return data, n_words, n_names

    def __len__(self):
        return len(self.data)
# DataUniqueWord:1 ends here

# ClsData


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*ClsData][ClsData:1]]
class ClsData(Dataset):
    """for classifier.
dataset: ncbi_ctd2, cdr_ctd2."""
    def __init__(self, data, batch_size):
        self.concept_id = {dt['concept']: idx
                           for idx, dt in zip(count(0), data)}
        self.data = [{"data": dt,
                      "n_words": n_w,
                      "label": self.concept_id[i['concept']],
                      "concept": i['concept']}
                     for i in data
                     for dt, n_w in zip(i['data'], i['n_words'])]
        self.len = len(self.data)
        self.idx = np.array(list(range(self.len)))
        self.batch_size = batch_size
        self.gen_data = self.gen_data_fn()
        self.gen_idx = self.gen_idx_fn()

    def gen_idx_fn(self):
        while True:
            np.random.shuffle(self.idx)
            for i in self.idx:
                yield i

    def gen_one_sample(self, dt):
        n_names = len(dt['n_words'])
        return [dt['data'], dt['n_words'], [n_names], dt['label']]

    def gen_data_fn(self):
        while True:
            batch = [self.data[self.gen_idx.__next__()]
                         for i in range(self.batch_size)]
            data = np.concatenate([i['data'] for i in batch])
            n_words = [i['n_words'] for i in batch]
            n_names = [1] * len(n_words)
            labels = [i['label'] for i in batch]
            n_words, n_names, labels = [np.array(i) for i in [n_words, n_names, labels]]
            yield data, n_words, n_names, labels

    def __getitem__(self, index):
        data, n_words, n_names, labels = self.gen_data.__next__()
        return data, n_words, n_names, labels

    def __len__(self):
        return self.len
# ClsData:1 ends here

# DataTokenized


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*DataTokenized][DataTokenized:1]]
class DataTokenized(Dataset):
    def __init__(self, data, batch_size, max_size_for_each_concept, zero_pad=-1, recode=True, classify=False):
        self.data = list(data.values())
        self.len = len(self.data)
        self.idx = np.array(range(self.len))
        self.batch_size = batch_size
        self.max_size_for_each_concept = max_size_for_each_concept
        self.gen_idx = self.gen_idx_fn()
        self.pad = zero_pad - len(self.data[0][0]['char_code'][0])
        self.gen_data = [self.gen_data_fn_not_pad, self.gen_data_fn,
                         self.gen_data_fn_pad][np.sum(self.pad >= np.array([0, 1]))]
        self.gen_data = self.gen_data if recode else self.gen_data_not_recode
        self.yield_data = self.yield_data_helper()
        if self.pad > 0:
            self.zero = np.int32(np.zeros([1000, self.pad]))
        if classify:
            self.concept_id = {c: idx for idx, c in enumerate(data)}
            self.gen_data = self.gen_data_classify

    def gen_idx_fn(self):
        while True:
            np.random.shuffle(self.idx)
            for i in self.idx:
                yield i

    def gen_dt_fn(self, dt):
        n_names = len(dt)
        if n_names <= self.max_size_for_each_concept:
            idx = range(n_names)
        else:
            idx = np.random.randint(
                0, high=n_names, size=self.max_size_for_each_concept)
            n_names = self.max_size_for_each_concept
        return [dt[i] for i in idx], n_names

    def gen_data_helper(self, dt, n_names, n_names_sum):
        new_dt, new_n_names = self.gen_dt_fn(self.data[self.gen_idx.__next__()])
        dt += new_dt
        n_names += [new_n_names]
        n_names_sum += new_n_names
        if n_names_sum < self.batch_size:
            return self.gen_data_helper(dt, n_names, n_names_sum)
        dd = defaultdict(list)
        for d in dt:
            for k, v in d.items():
                dd[k] += [v]
        return dict(dd), n_names

    def re_encode(self, word_code, char_code):
        word_list = list(chain(*word_code))
        word_unique = np.unique(word_list)
        # no pad_idx for word, so count(0)
        word_map = {k: v for k, v in zip(word_unique, count(0))}
        new_word_list = np.array([word_map[i] for i in word_list])
        word_code_map = {w: c for w, c in zip(chain(*word_code), chain(*char_code))}
        word_char_code = np.array([word_code_map[w] for w in word_unique])
        return word_char_code, new_word_list

    def gen_data_fn(self):
        dd, n_names = self.gen_data_helper([], [], 0)
        n_words = np.array(dd['n_word'])
        char_code, word_code = self.re_encode(dd["word_code"], dd['char_code'])
        return char_code, word_code, n_words, np.array(n_names)

    def gen_data_fn_not_pad(self):
        char_code, word_code, n_words, n_names = self.gen_data_fn()
        char_code = char_code[:, char_code.sum(0) > 0]
        return char_code, word_code, n_words, n_names

    def gen_data_fn_pad(self):
        char_code, word_code, n_words, n_names = self.gen_data_fn()
        char_code = np.hstack([char_code, self.zero[:char_code.shape[0]]])
        return char_code, word_code, n_words, n_names

    def gen_data_classify(self):
        dd, n_names = self.gen_data_helper([], [], 0)
        n_words = np.array(dd['n_word'])
        char_code, word_code = self.re_encode(dd["word_code"], dd['char_code'])
        concepts = dd["concept"]
        label = np.array([self.concept_id[c] for c in concepts])
        return char_code, word_code, n_words, np.array(n_names), label

    def gen_data_not_recode(self):
        dd, n_names = self.gen_data_helper([], [], 0)
        n_words = np.array(dd['n_word'])
        char_code = np.array(list(chain(*dd['char_code'])))
        word_code = np.array(list(chain(*dd['word_code'])))
        return char_code, word_code, n_words, np.array(n_names)

    def yield_data_helper(self):
        while True:
            yield self.gen_data()

    def __getitem__(self, index):
        return self.yield_data.__next__()

    def __len__(self):
        return self.len
# DataTokenized:1 ends here

# DataTokenizedWordPad


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/data_loader.org::*DataTokenizedWordPad][DataTokenizedWordPad:1]]
class DataTokenizedWordPad(Dataset):
    def __init__(self, data, batch_size, max_size_for_each_concept, zero_pad=-1):
        self.data = list(data.values())
        self.len = len(self.data)
        self.idx = np.array(range(self.len))
        self.batch_size = batch_size
        self.max_size_for_each_concept = max_size_for_each_concept
        self.gen_idx = self.gen_idx_fn()
        self.pad = zero_pad - len(self.data[0][0]['char_code'][0])
        self.gen_data = [self.gen_data_fn_not_pad, self.gen_data_fn,
                         self.gen_data_fn_pad][np.sum(self.pad >= np.array([0, 1]))]
        self.yield_data = self.yield_data_helper()
        if self.pad > 0:
            self.zero = np.int32(np.zeros([1000, self.pad]))

    def gen_idx_fn(self):
        while True:
            np.random.shuffle(self.idx)
            for i in self.idx:
                yield i

    def gen_dt_fn(self, dt):
        n_names = len(dt)
        if n_names <= self.max_size_for_each_concept:
            idx = range(n_names)
        else:
            idx = np.random.randint(
                0, high=n_names, size=self.max_size_for_each_concept)
            n_names = self.max_size_for_each_concept
        return [dt[i] for i in idx], n_names

    def gen_data_helper(self, dt, n_names, n_names_sum):
        new_dt, new_n_names = self.gen_dt_fn(self.data[self.gen_idx.__next__()])
        dt += new_dt
        n_names += [new_n_names]
        n_names_sum += new_n_names
        if n_names_sum < self.batch_size:
            return self.gen_data_helper(dt, n_names, n_names_sum)
        dd = defaultdict(list)
        for d in dt:
            for k, v in d.items():
                dd[k] += [v]
        return dict(dd), n_names

    def re_encode(self, word_code, char_code):
        word_list = np.array(word_code)
        word_list_p = word_list < word_list.max()
        word_list = word_list[:, word_list_p.sum(0) > 0]
        # word_list = list(chain(*word_code))
        word_unique = np.unique(word_list)
        # no pad_idx for word, so count(0)
        word_map = {k: v for k, v in zip(word_unique, count(0))}
        new_word_code = np.array([[word_map[j] for j in i] for i in word_list])
        word_code_map = {w: c for W, C in zip(word_code, char_code)
                         for w, c in zip(W, C)}
        # word_unique[:-1] to avoid word_pad
        new_char_code = np.array([word_code_map[w] for w in word_unique[:-1]])
        return new_word_code, new_char_code

    def gen_data_fn(self):
        dd, n_names = self.gen_data_helper([], [], 0)
        n_words = np.array(dd['n_word'])
        word_code, char_code = self.re_encode(dd["word_code"], dd['char_code'])
        return char_code, word_code, n_words, np.array(n_names)

    def gen_data_fn_not_pad(self):
        char_code, word_code, n_words, n_names = self.gen_data_fn()
        char_code = char_code[:, char_code.sum(0) > 0]
        return char_code, word_code, n_words, n_names

    def gen_data_fn_pad(self):
        char_code, word_code, n_words, n_names = self.gen_data_fn()
        char_code = np.hstack([char_code, self.zero[:char_code.shape[0]]])
        return char_code, word_code, n_words, n_names

    def yield_data_helper(self):
        while True:
            yield self.gen_data()

    def __getitem__(self, index):
        return self.yield_data.__next__()

    def __len__(self):
        return self.len
# DataTokenizedWordPad:1 ends here
