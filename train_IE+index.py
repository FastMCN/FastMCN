import os
import sys
import csv
import torch
import pickle
import numpy as np
import random
from pathlib import Path
from itertools import count, chain
from functools import partial, reduce
from importlib import import_module
from tqdm import trange
from utils.utils import *
from utils.data_loader import DataTokenized
from utils.loss_func import loss_func_set
from utils.model import model_set
from torch import optim, nn
from time import time
from datetime import datetime
from torch.nn.functional import normalize
from collections import defaultdict
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def test_warn_start(test_code, test_n_words):
    warm_start = time()
    test_ft = eval_get_ft.get_ft_fast(test_code, test_n_words)
    test_pred = (test_ft @ train_ft).argmax(-1).tolist()
    test_acc = np.mean([l == train_label[p]
                        for l, p in zip(test_label, test_pred)])
    return time() - warm_start


def dump_config(obj_name: str, path: str, dt: str):
    import json
    obj = globals()[obj_name]
    obj['start_time'] = dt
    tmp = {k: str(v) for k, v in obj.items()}
    with open(f"{path}/{obj_name}.json", "a") as f:
        json.dump(tmp, f, indent=4)
        f.write("\n")


class TrainRecGetdFt():
    def __init__(self, model, char_code, device):
        self.char_code = char_code
        self.model = model
        self.n_chars = (self.char_code > 0).sum(-1)
        self.word_idx = torch.tensor(list(range(self.char_code.shape[0])),
                                     device=device)
        self.qkv = model.compute_qkv()

    def __call__(self, word_lens, gt_word_len, word_ft, word_idx):
        intake = (self.n_chars <= word_lens[0]) * gt_word_len
        new_gt_word_len = self.n_chars > word_lens[0]
        char_code_intake = self.char_code[intake]
        char_code_intake = char_code_intake[:, char_code_intake.sum(0) > 0]
        word_idx += [self.word_idx[intake]]
        # word_ft += [self.model(
        #     char_code_intake, self.model.compute_qkv(char_code_intake.shape[-1]))]
        word_ft += [self.model(char_code_intake, self.qkv, char_code_intake.shape[-1])]
        if len(word_lens) > 1:
            return self.__call__(word_lens[1:], new_gt_word_len, word_ft, word_idx)
        _, word_idx_order = torch.cat(word_idx).sort()
        word_ft = torch.cat(word_ft).index_select(0, word_idx_order)
        return word_ft


class GetFt():
    def __init__(self, model, word_map, batch_size, device):
        self.model = model
        self.word_map = word_map
        self.batch_size = batch_size
        self.device = device
        self.vocab_map, self.vocab_char_codes = self.organize_word_map()
        self.vocab_char_codes = torch.tensor(
            self.vocab_char_codes, device=self.device)
        self.vocab_size = self.vocab_char_codes.shape[0]
        with torch.no_grad():
            self.qkv = model.compute_qkv()
            self.vocab_ft = self.get_word_ft(self.vocab_char_codes)

    def organize_word_map_helper(self, collecitons, dt):
        word_map, char_codes = collecitons
        idx, (key, char_code) = dt
        word_map[key] = idx
        char_codes += [char_code]
        return (word_map, char_codes)

    def organize_word_map(self):
        tmp_word_map = [(len(k), k, v) for k, v in self.word_map.items()]
        tmp_word_map = sorted(tmp_word_map)
        tmp_word_map = {i[1]: i[2] for i in tmp_word_map}
        word_map, char_codes = reduce(
            self.organize_word_map_helper,
            zip(count(0), tmp_word_map.items()),
            ({}, []))
        return word_map, char_codes

    def get_word_ft_helper(self, char_codes):
        char_codes = char_codes[:, char_codes.sum(0) > 0]
        word_ft = self.model(char_codes, self.qkv, char_codes.shape[-1])
        return word_ft

    def get_word_ft(self, char_codes):
        char_code_size = char_codes.shape[0]
        lower = list(range(0, char_code_size, self.batch_size))[:-1]
        upper = lower[1:] + [char_code_size]
        with torch.no_grad():
            word_ft = torch.cat([self.get_word_ft_helper(char_codes[l:u])
                                 for l, u in zip(lower, upper)])
        return word_ft

    def get_name_ft_helper(self, collecitons, dt):
        word_code, n_words, names, label = collecitons
        word_code += [self.vocab_map[w] for w in dt['word']]
        n_words += [dt['n_word']]
        names += [dt['name']]
        label += [dt['concept'].lower()]
        return (word_code, n_words, names, label)

    def get_data(self, data_set):
        word_code, n_words, names, label = reduce(
            self.get_name_ft_helper,
            chain(*data_set.values()),
            ([], [], [], []))
        word_code = torch.tensor(word_code, device=self.device)
        return word_code, n_words, names, label

    def __call__(self, word_code, n_words):
        word_ft = self.vocab_ft.index_select(0, word_code).split(n_words)
        name_ft = normalize(torch.stack([ft.mean(0) for ft in word_ft]))
        return name_ft

    def get_ft_fast(self, word_code, n_words_tensor):
        word_ft = self.vocab_ft.index_select(0, word_code)
        name_ft = normalize(torch.spmm(n_words_tensor, word_ft))
        return name_ft

    def get_data_fast(self, data_set):
        word_code, n_words, names, label = reduce(
            self.get_name_ft_helper,
            chain(*data_set.values()),
            ([], [], [], []))
        word_code = torch.tensor(word_code, device=self.device)
        sparse_idx = []
        sparse_value = []
        n = 0
        for i, w in enumerate(n_words):
            sparse_idx += [[i, c] for c in range(n, (n + w))]
            sparse_value += [1] * w
            n += w
        n_words_tensor = torch.sparse.FloatTensor(torch.tensor(sparse_idx).T,
                                                  torch.FloatTensor(sparse_value),
                                                  torch.Size([len(n_words), sum(n_words)])).to(self.device)
        return word_code, n_words_tensor, names, label


class YieldData():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.yield_data = self.yield_data_fn()

    def yield_data_fn(self):
        while True:
            for data in self.data_loader:
                yield data

    def __call__(self):
        return self.yield_data.__next__()


config_name = sys.argv[1] if len(sys.argv) > 1 else "cdr_svtransformer_bpetoken_v2_512"

config_vars = import_module(f"config.{config_name}")
config_var_names = [i for i in config_vars.__dir__() if not i.startswith("__")]

for n in config_var_names:
    globals()[n] = getattr(config_vars, n)

# NOTE: This will cover INIT_STEP and continue_config variable that
# are assigned at the config file.
if len(sys.argv) > 2:
    SEED = int(sys.argv[2])
    work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_{str(mlp_dims)}_{activate_func}_trial{SEED}_svt_2/"
    train_log = f"{work_dir}/train.log"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(GPU_NUM)

for name in VAR_NAMES:
    globals()[name] = load_by_name(DATA_PATH, name)

concept_id = {n.lower(): idx for n, idx in zip(train_data, count(0))}

dev_label = list(chain(*[[i['concept'].lower() for i in v] for v in dev_data.values()]))
test_label = list(chain(*[[i['concept'].lower() for i in v] for v in test_data.values()]))

model_config['args']['vocab_size'] = len(char_map) + 1

path_work_dir = Path(work_dir)

if not path_work_dir.exists():
    path_work_dir.mkdir(parents=True)

CSV_VARS = ["time", "var", "value"]

if not os.path.exists(train_log):
    with open(train_log, "w") as f:
        csv_writer = csv.DictWriter(
            f, CSV_VARS
        )
        csv_writer.writeheader()

formater = defaultdict(lambda: "%s: %.4f")
formater["step"] = "%s: %-4d"
formater["lr"] = "%s: %-4.1e"

optim_set = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}

start_time = now()

loss_func = loss_func_set[loss_func_config['name']]

if loss_func_config['args']:
    loss_func = partial(loss_func, **loss_func_config['args'])

model_cls = model_set[model_config['name']]

if state == "new":
    model = model_cls(**model_config['args'])
    if WEIGHT_INIT:
        weight_names = [i for i in model.state_dict().keys() if "weight" in i and "embedding" not in i]
        for n in weight_names:
            w = reduce(getattr, n.split("."), model)
            if len(w.shape) > 1:
                nn.init.kaiming_uniform_(w, nonlinearity="relu")
    optimizer = optim_set[optim_config['name']](
        model.parameters(), **optim_config['args'])
elif state == "continue":
    model = torch.load(f"{continue_config['model']}")
    optimizer = optim_set[optim_config['name']](
        model.parameters(), **optim_config['args'])
    if not NEW_OPTIMIZER:
        optimizer.load_state_dict(
            torch.load(f"{continue_config['optimizer']}").state_dict()
        )
        print(f"use optimizer: {continue_config['optimizer']}\n")
    configs.append("continue_config")
    print(f"continue training base on step {INIT_STEP}")
    print(f"use model: {continue_config['model']}")


model_size = np.sum([i.numel() * 4 for i in model.parameters()]) / 1024 / 1024
print(f"model_size: {model_size:.2f} Mib")

print(f"Everything will be stored at:\n{work_dir}")

dump_config_by_name = partial(dump_config, path=work_dir, dt=start_time)

for config in configs:
    dump_config_by_name(config)

train_loader = DataLoader(
    DataTokenized(train_data, BATCH_SIZE, MAX_SIZE_FOR_EACH_CONCEPT),
    num_workers=2,
    pin_memory=True)
yield_train_data = YieldData(train_loader)

if DEVICE == "cuda":
    model = model.cuda()
    cudnn.benchmark = True

print(model)
step_count = count(INIT_STEP + 1)
best_test_acc = 0
best_dev_acc = 0
# all_emb_index = model.all_emb_index

for _ in range(N_ROUNDS):
    step = step_count.__next__()
    start = time()
    cache = defaultdict(list)
    t = trange(STEP_PRE_ROUND, leave=False, ncols=120, ascii=True)
    for _ in t:
        char_code, word_code, n_words, n_names = [i.squeeze(0) for i in yield_train_data()]
        char_code, word_code = [i.to(DEVICE) for i in [char_code, word_code]]
        n_words, n_names = [list(i.numpy()) for i in [n_words, n_names]]
        train_rec_get_ft = TrainRecGetdFt(model, char_code, DEVICE)
        word_ft = train_rec_get_ft(WORD_LENS, True, [], [])
        word_ft = word_ft.index_select(0, word_code)
        word_ft = word_ft.split(n_words, dim=0)
        output = torch.stack([i.mean(0) for i in word_ft]).split(n_names)
        result = loss_func(output)
        if torch.isnan(result['loss']):
            break
            restart(step - 1)
        for param in model.parameters():
            param.grad = None
        # optimizer.zero_grad()
        result['loss'].backward()
        optimizer.step()
        result = {k: v.item() for k, v in result.items()}
        t.set_postfix(**result)
        for key, value in result.items():
            cache[key].append(value)
        cache['n_concepts'].append(len(n_names))
    metrics = {k: np.mean(v) for k, v in cache.items()}
    metrics['step'] = step
    metrics['eval_time'] = time() - start
    metrics['lr'] = optim_config['args']['lr']
    print(" ".join([formater[k] % (k, v) for k, v in metrics.items()]))
    metrics['batch_size'] = BATCH_SIZE
    current_time = now()
    with open(train_log, "a") as f:
        logs = [{"time": current_time,
                 "var": k,
                 "value": v}
                for k, v in metrics.items()]
        csv_writer = csv.DictWriter(
            f, CSV_VARS
        )
        csv_writer.writerows(logs)
    if step % EVAL_PRE_STEP == 0 and EVAL == True:
        cold_start = time()
        model.eval()
        eval_get_ft = GetFt(model, word_map, EVAL_BATCH_SIZE, DEVICE)
        train_code, train_n_words, train_names, train_label = eval_get_ft.get_data_fast(train_data)
        train_ft = eval_get_ft.get_ft_fast(train_code, train_n_words)
        train_ft = train_ft.transpose(0, 1)
        test_code, test_n_words, test_names, test_label = eval_get_ft.get_data_fast(test_data)
        warm_start = time()
        test_ft = eval_get_ft.get_ft_fast(test_code, test_n_words)
        # test_pred = list((test_ft @ train_ft).argmax(-1).to("cpu").numpy())
        test_pred = (test_ft @ train_ft).argmax(-1).tolist()
        eval_end = time()
        cold_start_time, warm_start_time = [eval_end - t for t in [cold_start, warm_start]]
        test_acc = np.mean([l == train_label[p]
                            for l, p in zip(test_label, test_pred)])
        dev_code, dev_n_words, dev_names, dev_label = eval_get_ft.get_data_fast(dev_data)
        dev_ft = eval_get_ft.get_ft_fast(dev_code, dev_n_words)
        dev_pred = (dev_ft @ train_ft).argmax(-1).tolist()
        dev_acc = np.mean([l == train_label[p]
                           for l, p in zip(dev_label, dev_pred)])
        print(f"dev acc: {dev_acc:.4f} test acc: {test_acc:.4f} cold_start: {cold_start_time:.4f} warm_start: {warm_start_time:.4f}")
        with open(f"{work_dir}/eval_log.txt", "a") as f:
            f.write(f"{step},{current_time},test_acc,{test_acc}\n")
            f.write(f"{step},{current_time},dev_acc,{dev_acc}\n")
            f.write(f"{step},{current_time},cold_start_time,{cold_start_time}\n")
            f.write(f"{step},{current_time},warm_start_time,{warm_start_time}\n")
        if test_acc > best_test_acc:
            torch.save(model, f"{work_dir}/{optim_config['name']}_best_test_model.pkl")
            best_test_acc = test_acc
            content = "\n".join(["\t".join([n, l, train_label[p]])
                                 for p, n, l in zip(test_pred, test_names, test_label)])
            with open(f"{work_dir}/best_test_acc_pred.tsv", "w") as f:
                f.write(content)
        if dev_acc > best_dev_acc:
            torch.save(model, f"{work_dir}/{optim_config['name']}_best_dev_model.pkl")
            best_dev_acc = dev_acc
            content = "\n".join(["\t".join([n, l, train_label[p]])
                                 for p, n, l in zip(test_pred, test_names, test_label)])
            with open(f"{work_dir}/best_dev_acc_pred.tsv", "w") as f:
                f.write(content)
        train_ft = test_ft = torch.tensor([0], device=DEVICE)
        torch.cuda.empty_cache()
        model.train()
    if SAVE_PRE_STEP and step % SAVE_PRE_STEP == 0:
        torch.save(model, f"{work_dir}/{optim_config['name']}_model_checkpoint.pkl")
        torch.save(optimizer, f"{work_dir}/{optim_config['name']}_checkpoint.pkl")


train_ft = eval_get_ft.get_ft_fast(train_code, train_n_words)
train_ft = train_ft.transpose(0, 1)
test_ft = eval_get_ft.get_ft_fast(test_code, test_n_words)
warn_start_time = [test_warn_start(test_code, test_n_words) for i in range(100)]
warn_start_time_mean = np.mean(warn_start_time)
print(f"Average time of warn_start: {warn_start_time_mean:.3f}")
print(f"Average time of one query: {(warn_start_time_mean * 1000 / test_ft.shape[0]):.3f} us")
