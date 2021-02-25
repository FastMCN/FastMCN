import os
import sys
import csv
import torch
import pickle
import random
import numpy as np
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
from model.transformer.transformer import Encoder, NameEncoder
from time import time
from datetime import datetime
from torch.nn.functional import normalize
from collections import defaultdict
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

print(sys.argv)


def dump_config(obj_name: str, path: str, dt: str):
    import json
    obj = globals()[obj_name]
    obj['start_time'] = dt
    tmp = {k: str(v) for k, v in obj.items()}
    with open(f"{path}/{obj_name}.json", "a") as f:
        json.dump(tmp, f, indent=4)
        f.write("\n")


class GetFt():
    def __init__(self, model, word_map, batch_size, device):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def get_word_ft(self, char_codes):
        char_code_size = char_codes.shape[0]
        lower = list(range(0, char_code_size, self.batch_size))[:-1]
        upper = lower[1:] + [char_code_size]
        word_ft = torch.cat([self.model(char_codes[l:u], char_codes.shape[-1])
                             for l, u in zip(lower, upper)])
        return word_ft

    def get_name_ft_helper(self, collecitons, dt):
        char_code, n_words, names, label = collecitons
        char_code += dt['char_code']
        n_words += [dt['n_word']]
        names += [dt['name']]
        label += [dt['concept'].lower()]
        return (char_code, n_words, names, label)

    def __call__(self, data_set):
        char_code, n_words, names, label = reduce(
            self.get_name_ft_helper,
            chain(*data_set.values()),
            ([], [], [], []))
        char_code = torch.tensor(char_code, device=self.device)
        with torch.no_grad():
            word_ft = self.get_word_ft(char_code).split(n_words)
        name_ft = normalize(torch.stack([ft.mean(0) for ft in word_ft]))
        return name_ft, names, label


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


config_name = sys.argv[1] if len(
    sys.argv) > 1 else "cdr_char_transformer_no_word_index_512"

config_vars = import_module(f"config.{config_name}")
config_var_names = [i for i in config_vars.__dir__() if not i.startswith("__")]

for n in config_var_names:
    globals()[n] = getattr(config_vars, n)


# NOTE: This will cover INIT_STEP and continue_config variable that
# are assigned at the config file.
if len(sys.argv) > 2:
    SEED = int(sys.argv[2])
    work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_trial{SEED}_svt/"
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

dev_label = list(chain(*[[i['concept'].lower() for i in v]
                         for v in dev_data.values()]))
test_label = list(chain(*[[i['concept'].lower()
                           for i in v] for v in test_data.values()]))

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
        weight_names = [i for i in model.state_dict(
        ).keys() if "weight" in i and "embedding" not in i]
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
    DataTokenized(train_data, BATCH_SIZE,
                  MAX_SIZE_FOR_EACH_CONCEPT, recode=False),
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
    # for _ in range(STEP_PRE_ROUND):
    for _ in t:
        char_code, word_code, n_words, n_names = [
            i.squeeze(0) for i in yield_train_data()]
        char_code, word_code = [i.to(DEVICE) for i in [char_code, word_code]]
        n_words, n_names = [i.tolist() for i in [n_words, n_names]]
        char_code = char_code[:, char_code.sum(0) > 0]
        max_len = char_code.shape[-1]
        word_ft = model(char_code, max_len)
        word_ft = word_ft.split(n_words, dim=0)
        output = torch.stack([i.mean(0) for i in word_ft]).split(n_names)
        result = loss_func(output)
        if torch.isnan(result['loss']):
            restart(step - 1)
        for param in model.parameters():
            param.grad = None
        # optimizer.zero_grad()
        result['loss'].backward()
        optimizer.step()
        del output, word_ft
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
        # eval_get_train_ft = GetTrainFt(model, word_map, EVAL_BATCH_SIZE, DEVICE)
        train_ft, train_names, train_label = eval_get_ft(train_data)
        train_ft = train_ft.transpose(0, 1)
        warm_start = time()
        test_ft, test_names, test_label = eval_get_ft(test_data)
        test_pred = list((test_ft @ train_ft).argmax(-1).to("cpu").numpy())
        eval_end = time()
        cold_start_time, warm_start_time = [
            eval_end - t for t in [cold_start, warm_start]]
        test_acc = np.mean([l == train_label[p]
                            for l, p in zip(test_label, test_pred)])
        dev_ft, dev_names, dev_label = eval_get_ft(dev_data)
        dev_pred = list((dev_ft @ train_ft).argmax(-1).to("cpu").numpy())
        dev_acc = np.mean([l == train_label[p]
                           for l, p in zip(dev_label, dev_pred)])
        print(
            f"dev acc: {dev_acc:.4f} test acc: {test_acc:.4f} cold_start: {cold_start_time:.4f} warm_start: {warm_start_time:.4f}")
        with open(f"{work_dir}/eval.log", "a") as f:
            f.write(f"{step},{current_time},test_acc,{test_acc}\n")
            f.write(f"{step},{current_time},dev_acc,{dev_acc}\n")
            f.write(f"{step},{current_time},cold_start_time,{cold_start_time}\n")
            f.write(f"{step},{current_time},warm_start_time,{warm_start_time}\n")
        if test_acc > best_test_acc:
            torch.save(
                model, f"{work_dir}/{optim_config['name']}_best_model.pkl")
            best_test_acc = test_acc
            content = "\n".join(["\t".join([n, l, train_label[p]])
                                 for p, n, l in zip(test_pred, test_names, test_label)])
            with open(f"{work_dir}/best_test_acc_pred.tsv", "w") as f:
                f.write(content)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            content = "\n".join(["\t".join([n, l, train_label[p]])
                                 for p, n, l in zip(test_pred, test_names, test_label)])
            with open(f"{work_dir}/best_dev_acc_pred.tsv", "w") as f:
                f.write(content)
        train_ft = test_ft = torch.tensor([0], device=DEVICE)
        torch.cuda.empty_cache()
        model.train()
    if SAVE_PRE_STEP and step % SAVE_PRE_STEP == 0:
        torch.save(
            model, f"{work_dir}/{optim_config['name']}_model_checkpoint.pkl")
        torch.save(
            optimizer, f"{work_dir}/{optim_config['name']}_checkpoint.pkl")


# Local Variables:
# elpy-shell-starting-directory: current-directory
# pyvenv-activate: "/home/chongliang/miniconda3/envs/disease_normalization/"
# End:
