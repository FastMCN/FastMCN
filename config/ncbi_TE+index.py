# train with train_use_bpe_token_v2.py

import torch
from numpy import pi

# new, continue
INIT_STEP = 0
SEED = 2                        # 1, 2, 3

WEIGHT_INIT = False

state = "new" if INIT_STEP == 0 else "continue"

DATA = "ncbi_ctd_without_consistent5_32768"
# DATA = "ncbi_ctd_tokenized_32768"
MAX_CHARS = 32

SAVE_PRE_STEP = None
EVAL_PRE_STEP = 1
EVAL_BATCH_SIZE = 256

BATCH_SIZE = 64
WORD_VEC_D = 512
N_ROUNDS = 250
STEP_PRE_ROUND = 100
MAX_SIZE_FOR_EACH_CONCEPT = 2
DATA_PATH = f"./proc_data/{DATA}/"
EVAL = True
GPU_NUM = ["0"]

DEVICE = "cuda"

WORD_LENS = [8, MAX_CHARS]

# for continue
NEW_OPTIMIZER = False

# continue_config will automatically append to configs according to state
configs = ['model_config', "loss_func_config", "optim_config"]

loss_func_config = {
    "name": "v1_6",
    "args": {
        "eye_mask": (torch.eye(BATCH_SIZE + 10) == 0).to(DEVICE),
        # "upper": torch.tensor(0.99).float().to("cuda"),
        "eps": torch.tensor(1e-4).float().to("cuda")
    }
}

optim_config = {
    "name": "adam",
    "args": {
        "lr": 1e-4,
        "weight_decay": 5e-5
    }}

model_config = {
    "name": "CharTransformerV2",
    "args": {
        "n_head": 32,
        "embedding_dim": WORD_VEC_D,
        "max_chars": MAX_CHARS,
        "dropout": 0.1,
        }}

VAR_NAMES = ["train_data", "dev_data", "test_data", "char_map",
             "word_map", "global_var", "bpe_vocab", "bpe_vocab_idx"]

# work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_head{model_config['args']['n_head']}_{str(mlp_dims)}/"
work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_lr{optim_config['args']['lr']}_trial{SEED}/"

continue_config = {
    "model": f"{work_dir}{optim_config['name']}_model_{INIT_STEP}.pkl",
    "optimizer": f"{work_dir}{optim_config['name']}_{INIT_STEP}.pkl"
}

train_log = f"{work_dir}/train.log"
