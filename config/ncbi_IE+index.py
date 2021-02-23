# train with train_use_bpe_token_v2.py

import torch
from numpy import pi

SEED = 1                        # 1, 2, 3

# new, continue
INIT_STEP = 0

WEIGHT_INIT = False

state = "new" if INIT_STEP == 0 else "continue"

DATA = "ncbi_ctd_without_consistent5_32768"
MAX_CHARS = 32

SAVE_PRE_STEP = None
EVAL_PRE_STEP = 1
EVAL_BATCH_SIZE = 512

BATCH_SIZE = 64
WORD_VEC_D = 512
N_ROUNDS = 150
STEP_PRE_ROUND = 100
MAX_SIZE_FOR_EACH_CONCEPT = 2
DATA_PATH = f"./proc_data/{DATA}/"
EVAL = True
GPU_NUM = ["0"]

# # BATCH_SIZE: 130, 64
WORD_LENS = [8, MAX_CHARS]

DEVICE = "cuda"

# for continue
NEW_OPTIMIZER = False

# continue_config will automatically append to configs according to state
configs = ['model_config', "loss_func_config", "optim_config"]

loss_func_config = {
    "name": "v1_6",
    "args": {
        "eye_mask": (torch.eye(BATCH_SIZE + 10) == 0).to(DEVICE),
        # "upper": torch.tensor(0.99).float().to("cuda"),
        "eps": torch.tensor(1e-5).float().to("cuda")
    }
}

optim_config = {
    "name": "adam",
    "args": {
        "lr": 1e-5,
        "weight_decay": 5e-5
    }}

activate_func = "Tanh"

model_config = {
    "name": "IE",
    "args": {
        "n_head": 32,
        "word_vec_d": WORD_VEC_D,
        "max_chars": MAX_CHARS,
        "dropout": 0.1,
        "mlp_config": [[16, activate_func, False],
                       [1, activate_func, False]]
        # "mlp_config": [[16, "Tanh", True],
        #                [1, "Tanh", True]]
        }}

VAR_NAMES = ["train_data", "dev_data", "test_data", "char_map",
             "word_map", "global_var", "bpe_vocab", "bpe_vocab_idx"]

mlp_dims = [c[0] for c in model_config['args']['mlp_config']]
# work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_head{model_config['args']['n_head']}_{str(mlp_dims)}/"
work_dir = f"./train/{DATA}/{model_config['name']}/bs{BATCH_SIZE}_vec{WORD_VEC_D}_{str(mlp_dims)}_{activate_func}_trial{SEED}_svt_2/"
# work_dir = "/tmp/ncbi/"


continue_config = {
    "model": f"{work_dir}{optim_config['name']}_model_{INIT_STEP}.pkl",
    "optimizer": f"{work_dir}{optim_config['name']}_{INIT_STEP}.pkl"
}

train_log = f"{work_dir}/train.log"
