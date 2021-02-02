# import


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*import][import:1]]
import torch
import numpy as np
from utils.utils import *
from utils import activate
from torch import nn
from torch.nn.functional import normalize
from model.svtransformer.layers import *

model_set = {}
# import:1 ends here

# Dense


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*Dense][Dense:1]]
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, activate_func, bias=True):
        """A linear layer and an activate function.
input_dim: Int. The dimension of input.
output_dim: Int.  The dimension of output.
activate_func: str. Activate function which were defined at utils.activate.
bias: Bool. bias argument for linear layer."""
        super().__init__()
        args = locals()
        args_names = [key for key in args if key != "self"]
        for n in args_names:
            setattr(self, n, args[n])
        # self.act_func = torch.jit.script(getattr(activate, activate_func)())
        self.act_func = getattr(activate, activate_func)() if activate_func else self.id
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def id(self, inputs):
        return inputs

    def forward(self, inputs):
        return self.act_func(self.linear(inputs))
# Dense:1 ends here

# CharNameTransformer

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*CharNameTransformer][CharNameTransformer:1]]
class CharNameTransformer(nn.Module):
    def __init__(self, word_vec_d, max_words, word_encoder_args, name_encoder_args, device="cuda"):
        super().__init__()
        self.word_encoder = Encoder(**word_encoder_args)
        self.name_encoder = NameEncoder(**name_encoder_args)
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
            inputs_not_pad = (
                torch.logical_not(inputs_mask).unsqueeze(-1).repeat([1, 1, self.word_vec_d])
            )
        word_ft = self.word_encoder(inputs, inputs_mask)
        name_emb = (word_ft * inputs_not_pad).sum(-2) / inputs_not_pad.sum(-2)
        name_emb = name_emb.split(n_words)
        name_emb = torch.cat(
            [
                torch.cat([i, self.concept_pad[i.shape[0] :]])
                for i in name_emb
            ]
        ).reshape([len(n_words), self.max_words, self.word_vec_d])
        name_ft = self.name_encoder(name_emb)
        concept_not_pad = name_emb != 0
        name_ft = (name_ft * concept_not_pad).sum(-2) / concept_not_pad.sum(-2)
        output = name_ft.split(n_names)
        return output

model_set['CharNameTransformer'] = CharNameTransformer
# CharNameTransformer:1 ends here

# WidthCharNameTransformer

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharNameTransformer][WidthCharNameTransformer:1]]
class WidthCharNameTransformer(nn.Module):
    def __init__(self, word_vec_d, max_words, word_encoder_args, name_encoder_args, device="cuda"):
        super().__init__()
        self.word_encoder = Encoder(**word_encoder_args)
        self.name_encoder = WidthNameEncoder(**name_encoder_args)
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
            inputs_not_pad = (
                torch.logical_not(inputs_mask).unsqueeze(-1).repeat([1, 1, self.word_vec_d])
            )
        word_ft = self.word_encoder(inputs, inputs_mask)
        name_emb = (word_ft * inputs_not_pad).sum(-2) / inputs_not_pad.sum(-2)
        name_emb = name_emb.split(n_words)
        name_emb = torch.cat(
            [
                torch.cat([i, self.concept_pad[i.shape[0] :]])
                for i in name_emb
            ]
        ).reshape([len(n_words), self.max_words, self.word_vec_d])
        name_ft = self.name_encoder(name_emb)
        concept_not_pad = (name_emb == 0).all(-1).unsqueeze(-1)
        name_ft = name_ft.masked_fill(concept_not_pad, 0)
        name_ft = name_ft.sum(-2) / (name_ft != 0).sum(-2)
        output = name_ft.split(n_names)
        return output

model_set['WidthCharNameTransformer'] = WidthCharNameTransformer
# WidthCharNameTransformer:1 ends here

# CharTransformer


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*CharTransformer][CharTransformer:1]]
class CharTransformer(nn.Module):
    def __init__(self, word_vec_d, max_words, word_encoder_args, device="cuda"):
        super().__init__()
        self.word_encoder = Encoder(**word_encoder_args)
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
            inputs_not_pad = (
                torch.logical_not(inputs_mask).unsqueeze(-1).repeat([1, 1, self.word_vec_d])
            )
        word_ft = self.word_encoder(inputs, inputs_mask)
        name_emb = (word_ft * inputs_not_pad).sum(-2) / inputs_not_pad.sum(-2)
        name_emb = name_emb.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        output = name_ft.split(n_names)
        return output

model_set['CharNameTransformer'] = CharTransformer
# CharTransformer:1 ends here

# CharTransformerV2


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*CharTransformerV2][CharTransformerV2:1]]
class CharTransformerV2(nn.Module):
    def __init__(self, n_src_vocab, max_chars, embedding_dim, n_head,
                 dropout, pad_idx=0, device="cuda"):
        super().__init__()
        self.n_src_vocab = n_src_vocab
        self.max_chars = max_chars,
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.device = device
        self.pos_emb = self.gen_pos_embedding(max_chars, self.embedding_dim).to(device)
        self.char_emb = nn.Embedding(num_embeddings=self.n_src_vocab,
                                     embedding_dim=self.embedding_dim,
                                     padding_idx=self.pad_idx)
        self.qkv_linear = nn.Linear(in_features=self.embedding_dim,
                                    out_features=3 * self.n_head * self.embedding_dim,
                                    bias=False)
        # mha: multi-head attention
        self.mha_temperature = self.embedding_dim ** 0.5
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_bn = nn.LayerNorm(self.embedding_dim, eps=1e-6)
        self.mha_fc = nn.Linear(in_features=self.n_head * self.embedding_dim,
                                out_features=self.embedding_dim,
                                bias=False)
        # self.mha_fc_dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(p=dropout)
        # fw: feed forward
        self.ff_linear1 = nn.Linear(in_features=self.embedding_dim,
                                    out_features=self.embedding_dim)
        self.ff_linear2 = nn.Linear(in_features=self.embedding_dim,
                                    out_features=self.embedding_dim)
        # self.ff_dropout = nn.Dropout(dropout)
        self.ff_bn = nn.LayerNorm(self.embedding_dim, eps=1e-6)

    def gen_pos_embedding(self, n_position, d_hid):
        d_hid_half = d_hid >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / d_hid_half).repeat([d_hid_half])
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, n_position)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([n_position, d_hid])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding.unsqueeze(0)

    def multi_head_attention(self, char_ft, mask):
        # (n_words, n_chars, emb_dim) -> (n_words, n_chars, 3, n_head, emb_dim) -> (3, n_words, n_head, n_chars, emb_dim)
        char_qkv = self.qkv_linear(char_ft).reshape(*char_ft.shape[:2], 3, self.n_head, self.embedding_dim).permute(2, 0, 3, 1, -1)
        # (3, n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, emb_dim)
        char_q, char_k, char_v = [char_qkv[i] for i in range(3)]
        # (n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, n_chars)
        attn_score = torch.einsum("whqd, whkd->whqk", char_q, char_k) / self.mha_temperature
        attn_score = attn_score.masked_fill(mask, -1e9)
        attn_score = self.mha_dropout(nn.functional.softmax(attn_score, dim=-1))
        # (n_words, n_head, n_chars, n_chars), (n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, emb_dim)
        output = torch.einsum("whqk,whvd->whqd", attn_score, char_v)
        # (n_words, n_head, n_chars, emb_dim) -> (n_words, n_chars, n_head, emb_dim) -> (n_words, n_chars, n_head * emb_dim)
        output = output.transpose(1, 2).reshape(*char_ft.shape[:2], -1)
        # (n_words, n_chars, n_head * emb_dim) -> (n_words, n_chars, emb_dim)
        output = self.mha_fc(output)
        # output = self.mha_fc_dropout(output)
        output = self.mha_bn(output + char_ft)
        return output

    def feed_forward(self, inputs):
        output = nn.functional.relu(self.ff_linear1(inputs))
        output = self.ff_linear2(output)
        # output = self.ff_dropout(output)
        output = self.ff_bn(output + inputs)
        return output

    def get_mask(self, inputs):
        # (n_words, n_chars) -> (n_words, 1, 1, n_chars)
        return (inputs == self.pad_idx).reshape(inputs.shape[0], 1, 1, inputs.shape[-1])

    def forward(self, inputs, max_len):
        # (n_words, n_chars) -> (n_words, 1, 1, n_chars)
        mask = self.get_mask(inputs)
        # (n_words, n_chars) -> (n_words, n_chars, emb_dim)
        char_ft = self.char_emb(inputs) + self.pos_emb[:, :max_len]
        # (n_words, n_chars, emb_dim) -> (n_words, n_chars, emb_dim)
        char_ft = self.multi_head_attention(char_ft, mask)
        char_ft = self.feed_forward(char_ft)
        # (n_words, 1, 1, n_chars) -> (n_words, 1, n_chars)
        not_mask = mask.logical_not().squeeze(-2).float()
        word_ft = (not_mask @ char_ft).squeeze(-2) / not_mask.sum(-1)
        return word_ft

model_set['CharTransformerV2'] = CharTransformerV2
# CharTransformerV2:1 ends here

# CharTransformerV3

# no feed forward


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*CharTransformerV3][CharTransformerV3:1]]
class CharTransformerV3(nn.Module):
    def __init__(self, n_src_vocab, max_chars, embedding_dim, n_head,
                 dropout, pad_idx=0, device="cuda"):
        super().__init__()
        self.n_src_vocab = n_src_vocab
        self.max_chars = max_chars,
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.device = device
        self.pos_emb = self.gen_pos_embedding(max_chars, self.embedding_dim).to(device)
        self.char_emb = nn.Embedding(num_embeddings=self.n_src_vocab,
                                     embedding_dim=self.embedding_dim,
                                     padding_idx=self.pad_idx)
        self.qkv_linear = nn.Linear(in_features=self.embedding_dim,
                                    out_features=3 * self.n_head * self.embedding_dim,
                                    bias=False)
        # mha: multi-head attention
        self.mha_temperature = self.embedding_dim ** 0.5
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_bn = nn.LayerNorm(self.embedding_dim, eps=1e-6)
        self.mha_fc = nn.Linear(in_features=self.n_head * self.embedding_dim,
                                out_features=self.embedding_dim,
                                bias=False)

    def gen_pos_embedding(self, n_position, d_hid):
        d_hid_half = d_hid >> 1
        freq = torch.pow(torch.tensor([1e4]), -1 / d_hid_half).repeat([d_hid_half])
        freq[0] = 1.0
        freq = freq.cumprod(-1)
        position = torch.arange(0, n_position)
        phase = torch.einsum("i, j->ij", position, freq)
        pos_embedding = torch.zeros([n_position, d_hid])
        pos_embedding[:, 0::2] = torch.sin(phase)
        pos_embedding[:, 1::2] = torch.cos(phase)
        return pos_embedding.unsqueeze(0)

    def multi_head_attention(self, char_ft, mask):
        # (n_words, n_chars, emb_dim) -> (n_words, n_chars, 3, n_head, emb_dim) -> (3, n_words, n_head, n_chars, emb_dim)
        char_qkv = self.qkv_linear(char_ft).reshape(*char_ft.shape[:2], 3, self.n_head, self.embedding_dim).permute(2, 0, 3, 1, -1)
        # (3, n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, emb_dim)
        char_q, char_k, char_v = [char_qkv[i] for i in range(3)]
        # (n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, n_chars)
        attn_score = torch.einsum("whqd, whkd->whqk", char_q, char_k) / self.mha_temperature
        attn_score = attn_score.masked_fill(mask, -1e9)
        attn_score = self.mha_dropout(nn.functional.softmax(attn_score, dim=-1))
        # (n_words, n_head, n_chars, n_chars), (n_words, n_head, n_chars, emb_dim) -> (n_words, n_head, n_chars, emb_dim)
        output = torch.einsum("whqk,whvd->whqd", attn_score, char_v)
        # (n_words, n_head, n_chars, emb_dim) -> (n_words, n_chars, n_head, emb_dim) -> (n_words, n_chars, n_head * emb_dim)
        output = output.transpose(1, 2).reshape(*char_ft.shape[:2], -1)
        # (n_words, n_chars, n_head * emb_dim) -> (n_words, n_chars, emb_dim)
        output = self.mha_fc(output)
        # output = self.mha_fc_dropout(output)
        # output = self.mha_bn(output + char_ft)
        otuput = nn.functional.tanh(output)
        return output

    def get_mask(self, inputs):
        # (n_words, n_chars) -> (n_words, 1, 1, n_chars)
        return (inputs == self.pad_idx).reshape(inputs.shape[0], 1, 1, inputs.shape[-1])

    def forward(self, inputs, max_len):
        # (n_words, n_chars) -> (n_words, 1, 1, n_chars)
        mask = self.get_mask(inputs)
        # (n_words, n_chars) -> (n_words, n_chars, emb_dim)
        char_ft = self.char_emb(inputs) + self.pos_emb[:, :max_len]
        # (n_words, n_chars, emb_dim) -> (n_words, n_chars, emb_dim)
        char_ft = self.multi_head_attention(char_ft, mask)
        # (n_words, 1, 1, n_chars) -> (n_words, 1, n_chars)
        not_mask = mask.logical_not().squeeze(-2).float()
        word_ft = (not_mask @ char_ft).squeeze(-2) / not_mask.sum(-1)
        return word_ft

model_set['CharTransformerV3'] = CharTransformerV3
# CharTransformerV3:1 ends here

# WidthCharTransformer


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformer][WidthCharTransformer:1]]
class WidthCharTransformer(nn.Module):
    """A multi head attention layer and a fc layer."""
    def __init__(self, word_vec_d, max_words, word_encoder_args, output_dim, device="cuda"):
        super().__init__()
        self.word_encoder = MHAEncoder(**word_encoder_args)
        self.output_dim = output_dim
        self.mha_dim = word_encoder_args['n_head'] * word_encoder_args['d_v']
        self.fc = nn.Linear(self.mha_dim, output_dim, bias=False)
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        name_emb = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        # name_emb = (word_ft * inputs_not_pad).sum(-2) / inputs_not_pad.sum(-2)
        name_emb = name_emb.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        name_ft = self.fc(name_ft)
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformer'] = WidthCharTransformer
# WidthCharTransformer:1 ends here

# WidthCharTransformerMLP


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLP][WidthCharTransformerMLP:1]]
class WidthCharTransformerMLP(nn.Module):
    """A multi head attention layer and a fc layer."""
    def __init__(self, word_vec_d, max_words, word_encoder_args, mlp_dims, device="cuda"):
        super().__init__()
        # attributes
        self.mlp_dims = mlp_dims
        self.mha_dim = word_encoder_args['n_head'] * word_encoder_args['d_v']
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)
        # layers
        self.word_encoder = MHAEncoder(**word_encoder_args)
        mlp_dims = [self.mha_dim] + mlp_dims
        self.mlp = nn.Sequential(*[
            Dense(i, o, "Tanh", bias=False)
            for i, o in zip(mlp_dims[:-1], mlp_dims[1:])])

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        name_emb = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        # name_emb = (word_ft * inputs_not_pad).sum(-2) / inputs_not_pad.sum(-2)
        name_emb = name_emb.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        name_ft = self.mlp(name_ft)
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLP'] = WidthCharTransformerMLP
# WidthCharTransformerMLP:1 ends here

# WidthCharTransformerMLPHeader


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeader][WidthCharTransformerMLPHeader:1]]
class WidthCharTransformerMLPHeader(nn.Module):
    """A multi head attention layer and a fc layer."""

    def __init__(self, word_vec_d, max_words, word_encoder_args, mlp_config, device="cuda"):
        super().__init__()
        # attributes
        self.mlp_config = mlp_config
        self.mha_dim = word_encoder_args['n_head']
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros(
                [self.max_words, self.word_vec_d]).to(self.device)
        # layers
        self.word_encoder = MHAEncoderNoFlatten(**word_encoder_args)
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])

    def get_word_ft(self, inputs):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        word_ft = word_ft.transpose(2, 3)
        word_ft = self.mlp(word_ft)
        # shape: n_words, n_chars, features
        word_ft = word_ft.reshape(list(word_ft.shape[:2]) + [-1])
        # mean
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        # shape: n_words, features
        word_ft = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        return word_ft

    def forward(self, inputs, n_words, n_names):
        word_ft = self.get_word_ft(inputs)
        word_ft = word_ft.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        output = name_ft.split(n_names)
        return output


model_set['WidthCharTransformerMLPHeader'] = WidthCharTransformerMLPHeader
# WidthCharTransformerMLPHeader:1 ends here

# WidthCharTransformerMLPHeader2


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeader2][WidthCharTransformerMLPHeader2:1]]
class WidthCharTransformerMLPHeader2(nn.Module):
    """Like WidthCharTransformerMLPHeader, but run activate function before MLP layers."""
    def __init__(self, word_vec_d, max_words, word_encoder_args, activate_func, mlp_config, device="cuda"):
        super().__init__()
        # attributes
        self.word_vec_d = word_vec_d
        self.max_words = max_words
        self.activate_func = getattr(activate, activate_func)()
        self.mlp_config = mlp_config
        self.mha_dim = word_encoder_args['n_head']
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)
        # layers
        self.word_encoder = MHAEncoderNoFlatten(**word_encoder_args)
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        word_ft = self.activate_func(word_ft.transpose(2, 3).contiguous())
        word_ft = self.mlp(word_ft)
        word_ft = word_ft.reshape(list(word_ft.shape[:2]) + [-1])
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        name_emb = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        name_emb = name_emb.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLPHeader2'] = WidthCharTransformerMLPHeader2
# WidthCharTransformerMLPHeader2:1 ends here

# WidthCharTransformerMLPHClassifier


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHClassifier][WidthCharTransformerMLPHClassifier:1]]
class WidthCharTransformerMLPHClassifier(nn.Module):
    """A multi head attention layer and a fc layer."""
    def __init__(self, word_vec_d, max_words, word_encoder_args, mlp_config, n_classes, device="cuda"):
        super().__init__()
        # attributes
        self.mlp_config = mlp_config
        self.mha_dim = word_encoder_args['n_head']
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        with torch.no_grad():
            self.concept_pad = torch.zeros([self.max_words, self.word_vec_d]).to(self.device)
        # layers
        self.word_encoder = MHAEncoderNoFlatten(**word_encoder_args)
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])
        self.output_layer = nn.Linear(self.word_vec_d * mlp_config[-1][0],
                                      n_classes)

    def forward(self, inputs, n_words, n_names):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        word_ft = word_ft.transpose(2, 3)
        word_ft = self.mlp(word_ft)
        word_ft = word_ft.reshape(list(word_ft.shape[:2]) + [-1])
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        name_emb = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        name_emb = name_emb.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in name_emb])
        output = self.output_layer(name_ft)
        return output

model_set['WidthCharTransformerMLPHClassifier'] = WidthCharTransformerMLPHClassifier
# WidthCharTransformerMLPHClassifier:1 ends here

# WidthCharTransformerWordSim


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerWordSim][WidthCharTransformerWordSim:1]]
class WidthCharTransformerWordSim(nn.Module):
    """A multi head attention layer, MLP layer, and compute cosine distance after weighted."""
    def __init__(self, word_vec_d, max_words, word_encoder_args, mlp_config, device="cuda"):
        super().__init__()
        # attributes
        self.mlp_config = mlp_config
        self.mha_dim = word_encoder_args['n_head']
        self.max_words = max_words
        self.word_vec_d = word_vec_d
        self.device = device
        self.zero_pad = torch.zeros([self.max_words, mlp_config[-1][0] * self.word_vec_d], device=self.device)
        # layers
        self.word_encoder = MHAEncoderNoFlatten(**word_encoder_args)
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims[:-1], mlp_config)])
        self.word_att_q = torch.nn.Linear(mlp_dims[-1] * self.word_vec_d,
                                          self.word_vec_d,)
        self.word_att_v = torch.nn.Linear(mlp_dims[-1] * self.word_vec_d,
                                          self.word_vec_d,)

    def get_word_ft(self, inputs):
        with torch.no_grad():
            inputs_mask = inputs == 0
        word_ft = self.word_encoder(inputs, inputs_mask)
        word_ft = word_ft.transpose(2, 3)
        word_ft = self.mlp(word_ft)
        # shape: n_words, n_chars, features
        word_ft = word_ft.reshape(list(word_ft.shape[:2]) + [-1])
        # mean
        mask = inputs_mask.unsqueeze(-2).logical_not().float()
        # shape: n_words, features
        word_ft = torch.matmul(mask, word_ft).squeeze(-2) / mask.sum(-1)
        return word_ft

    def split_and_padding(self, features, n_words):
        features = features.split(n_words)
        features = torch.stack([torch.cat([i, self.zero_pad[i.shape[0]:self.max_words]])
                                for i in features])
        return features

    def forward(self, inputs, n_words, n_names):
        word_ft = self.get_word_ft(inputs)
        word_ft_q = self.word_att_q(name_emb)
        word_ft_v = self.word_att_v(name_emb)
        word_ft, word_ft_q, word_ft_v = [split_and_padding(ft) for ft in [word_ft, word_ft_q, word_ft_v]]
        return {"word_ft": word_ft,
                "word_ft_q": word_ft_q,
                "word_ft_v": word_ft_v}

model_set['WidthCharTransformerWordSim'] = WidthCharTransformerWordSim
# WidthCharTransformerWordSim:1 ends here

# WidthCharTransformerMLPHeaderFast


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFast][WidthCharTransformerMLPHeaderFast:1]]
class WidthCharTransformerMLPHeaderFast(nn.Module):
    """Fast WidthCharTransformerMLPHeader."""
    def __init__(self, n_src_vocab, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__()
        # attributes
        self.n_src_vocab = n_src_vocab
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        # layers
        self.embedding = nn.Embedding(self.n_src_vocab, self.word_vec_d, padding_idx=0)
        self.char_linear_qkv = nn.Linear(self.word_vec_d, 3 * self.n_head * self.word_vec_d, bias=False)
        # sdpa: scaled dot product attention
        self.sdpa_temperature=self.word_vec_d ** 0.5
        self.sdpa_dropout = nn.Dropout(self.dropout)
        self.sdpa_pos = nn.Linear(self.word_vec_d, self.n_head * self.word_vec_d, bias=False)
        self.sdpa_bn_layer = nn.LayerNorm(self.word_vec_d, eps=1e-6)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.bn_layer = nn.LayerNorm(word_vec_d, eps=1e-6)
        # self.slf_attn_no_flatten = MultiHeadAttentionNoFlatten(n_head, word_vec_d, word_vec_d, word_vec_d, dropout=dropout)
        # layers
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])

    def get_pad_mask(self, inputs):
        return (inputs != self.pad_idx).unsqueeze(-2)

    def char_qkv(self):
        embedding_w = self.embedding.weight
        # embedding_w = self.embedding(self.all_emb_index)
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[-1]]
        # char_qkv shape: n_head, n_src_vocab, word_vec_d
        char_q, char_k, char_v = torch.split(
            self.char_linear_qkv(embedding_w).reshape(qkv_shape),
            [self.n_head] * 3, dim=-2)
        return char_q, char_k, char_v

    def scaled_dot_product_attention(self, inputs, char_qkv):
        char_q, char_k, char_v = char_qkv
        char_sim = char_q.transpose_(0, 1) @ char_k.transpose_(0, 1).transpose_(1, 2)
        char_sim = char_sim / self.sdpa_temperature
        # mask pad_idx
        char_sim[:, :, self.pad_idx] = -1e9
        # # This equals to
        # # [torch.cat([i.index_select(0, dt).unsqueeze_(0) for dt in inputs]) for i in [char_q, char_k, char_v]]
        # # but much faster.
        # inputs_qpk = [i.index_select(0, inputs.reshape(-1)).reshape([*inputs.shape, self.n_head, self.word_vec_d]) for i in [char_q, char_k, char_v]]
        # This equals to:
        # q, k = inputs_qpk[:2]
        # inputs_sim = (q.transpose(1, 2) / temperature) @ k.transpose(1, 2).transpose(2, 3)
        # but much faster.
        inputs_sim = torch.cat([char_sim.index_select(1, i).index_select(-1, i).unsqueeze_(0)
                                for i in inputs])
        inputs_attn = self.sdpa_dropout(nn.functional.softmax(inputs_sim, dim=-1))
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape([*inputs.shape, self.n_head, self.word_vec_d]).transpose_(1, 2)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose_(1, 2).contiguous()

        output = self.sdpa_bn_layer(output)
        return output

    def get_word_ft(self, inputs, mask):
        char_qkv = self.char_qkv()
        # char_ft shape: n_words, max_chars, word_vec_d, n_head
        char_ft = self.scaled_dot_product_attention(inputs, char_qkv).transpose_(2, 3)
        # char_ft shape: n_words, max_chars, word_vec_d, mlp_last_layer
        char_ft = self.mlp(char_ft)
        # shape: n_words, n_chars, features
        char_ft = char_ft.reshape(list(char_ft.shape[:-2]) + [-1])
        # shape: n_words, features
        word_ft = (mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)
        return word_ft

    def forward(self, inputs, n_words, n_names):
        mask = self.get_pad_mask(inputs)
        word_ft = self.get_word_ft(inputs, mask)
        word_ft = word_ft.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output


model_set['WidthCharTransformerMLPHeaderFast'] = WidthCharTransformerMLPHeaderFast
# WidthCharTransformerMLPHeaderFast:1 ends here

# WidthCharTransformerMLPHeaderFastPos


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPos][WidthCharTransformerMLPHeaderFastPos:1]]
class WidthCharTransformerMLPHeaderFastPos(nn.Module):
    """Fast WidthCharTransformerMLPHeader."""
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__()
        # attributes
        self.n_src_vocab = n_src_vocab
        self.max_chars = max_chars
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        # layers
        self.embedding = nn.Embedding(
            n_src_vocab, word_vec_d, padding_idx=pad_idx)
        self.pos = self.get_pos(max_chars).to(device)
        self.char_qkv_layer = nn.Linear(
            word_vec_d, 3 * n_head * word_vec_d, bias=False)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = word_vec_d ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        self.sdpa_linear_layer = nn.Linear(
            max_chars, self.pos.shape[-2], bias=False)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        # self.slf_attn_no_flatten = MultiHeadAttentionNoFlatten(n_head, word_vec_d, word_vec_d, word_vec_d, dropout=dropout)
        # layers
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])

    def get_pad_mask(self, inputs):
        return (inputs != self.pad_idx).unsqueeze(-2)

    def get_pos(self, max_len):
        phrase = torch.tensor([list(range(-i, max_len - i))
                               for i in range(max_len)])
        freq = torch.linspace(0, max_len, max_len)[1:int(np.ceil(max_len / 2))]
        pos = (freq.unsqueeze(-1).unsqueeze(-1) *
               phrase.unsqueeze(0)) / max_len * 2 * np.pi
        pos = torch.cat([torch.cos(pos), torch.sin(pos)])
        pos = pos.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        return pos

    def compute_qkv(self):
        embedding_w = self.embedding.weight
        # embedding_w = self.embedding(self.all_emb_index)
        qkv_shape = [embedding_w.shape[0], 3 *
                     self.n_head, embedding_w.shape[1]]
        # char_qkv shape: n_head, n_src_vocab, word_vec_d
        char_q, char_k, char_v = torch.split(self.char_qkv_layer(embedding_w).reshape(qkv_shape),
                                             [self.n_head] * 3, dim=-2)
        # char_qkv = [i(embedding_w).reshape(qkv_shape)
        #             for i in self.char_qkv_layers]
        return char_q, char_k, char_v

    def scaled_dot_product_attention(self, inputs, mask, qkv):
        char_q, char_k, char_v = qkv
        char_sim = char_q.transpose(
            0, 1) @ char_k.transpose(0, 1).transpose(1, 2)
        char_sim = char_sim / self.sdpa_temperature
        # mask pad_idx
        char_sim[:, :, self.pad_idx] = -1e9
        # # This equals to
        # # [torch.cat([i.index_select(0, dt).unsqueeze(0) for dt in inputs]) for i in [char_q, char_k, char_v]]
        # # but much faster.
        # inputs_qpk = [i.index_select(0, inputs.reshape(-1)).reshape([*inputs.shape, self.n_head, self.word_vec_d]) for i in [char_q, char_k, char_v]]
        # This equals to:
        # q, k = inputs_qpk[:2]
        # inputs_sim = (q.transpose(1, 2) / temperature) @ k.transpose(1, 2).transpose(2, 3)
        # but much faster.
        inputs_sim = torch.cat([char_sim.index_select(1, i).index_select(-1, i).unsqueeze(0)
                                for i in inputs])
        pos_w = self.sdpa_linear_layer(
            inputs_sim * (inputs_sim > -1e8)).unsqueeze(-1)
        pos_w = (pos_w * self.pos).mean(-2)
        inputs_sim = inputs_sim.add(pos_w)
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def get_word_ft(self, inputs, mask, qkv):
        # char_ft shape: n_words, max_chars, word_vec_d, n_head
        char_ft = self.scaled_dot_product_attention(inputs, mask, qkv).transpose(2, 3)
        # char_ft shape: n_words, max_chars, word_vec_d, mlp_last_layer
        char_ft = self.mlp(char_ft)
        # shape: n_words, n_chars, features
        char_ft = char_ft.reshape(list(char_ft.shape[:-2]) + [-1])
        # shape: n_words, features
        word_ft = (mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)
        return word_ft

    def forward(self, inputs, n_words, n_names, qkv):
        mask = self.get_pad_mask(inputs)
        word_ft = self.get_word_ft(inputs, mask, qkv)
        word_ft = word_ft.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output


model_set['WidthCharTransformerMLPHeaderFastPos'] = WidthCharTransformerMLPHeaderFastPos
# WidthCharTransformerMLPHeaderFastPos:1 ends here

# WidthCharTransformerMLPHeaderFastPosV2

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV2][WidthCharTransformerMLPHeaderFastPosV2:1]]
class WidthCharTransformerMLPHeaderFastPosV2(nn.Module):
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__()
        # attributes
        self.n_src_vocab = n_src_vocab
        self.max_chars = max_chars
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        # layers
        self.embedding = nn.Embedding(
            n_src_vocab, word_vec_d, padding_idx=pad_idx)
        self.pos = self.get_pos(max_chars).to(device)
        self.char_qkv_layer = nn.Linear(
            word_vec_d, 3 * n_head * word_vec_d, bias=False)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = word_vec_d ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout_layer = nn.Dropout(dropout)
        self.bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        # self.slf_attn_no_flatten = MultiHeadAttentionNoFlatten(n_head, word_vec_d, word_vec_d, word_vec_d, dropout=dropout)
        # layers
        mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(mlp_dims, mlp_config)])

    def get_pad_mask(self, inputs):
        return (inputs != self.pad_idx).unsqueeze(-2)

    def get_pos(self, max_len):
        freq = self.word_vec_d >> 1
        phrase = np.float32(np.outer(
            range(max_len), 1 / np.power(10000, np.array(range(freq)) / freq)))
        pos = torch.tensor(
            np.hstack([np.cos(phrase), np.sin(phrase)]), device=self.device)
        return pos.reshape([1, 1, 1, *pos.shape])

    def compute_qkv(self):
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3,
                     self.n_head, embedding_w.shape[1]]
        # char_qkv shape: n_head, n_src_vocab, word_vec_d
        char_qkv = self.char_qkv_layer(embedding_w).reshape(qkv_shape)
        return char_qkv

    def scaled_dot_product_attention(self, inputs, mask):
        char_qkv = self.char_qkv()
        # inputs q, k, v shape: n_words, 1, n_head, max_chars, word_vec_d
        inputs_q, inputs_k, inputs_v = torch.split(
            char_qkv.index_select(0, inputs.reshape(-1)).reshape(
                [*inputs.shape, 3, self.n_head, self.word_vec_d]).transpose(1, 2).transpose(2, 3).add(self.pos),
            [1] * 3, dim=1)
        # inputs q, k, v shape: n_words, n_head, max_chars, word_vec_d
        for i in [inputs_q, inputs_k, inputs_v]:
            i.squeeze(1)
        # inputs_sim shape: n_words, n_head, max_chars, max_chars
        inputs_sim = (inputs_q / self.sdpa_temperature) @ inputs_k.transpose(-1, -2)
        inputs_sim = inputs_sim.masked_fill(
            mask.unsqueeze(1) == self.pad_idx, -1e9)
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def get_word_ft(self, inputs, mask, qkv):
        # char_ft shape: n_words, max_chars, word_vec_d, n_head
        char_ft = self.scaled_dot_product_attention(inputs, mask, qkv).transpose(2, 3)
        # char_ft shape: n_words, max_chars, word_vec_d, mlp_last_layer
        char_ft = self.mlp(char_ft)
        # shape: n_words, n_chars, features
        char_ft = char_ft.reshape(list(char_ft.shape[:-2]) + [-1])
        # shape: n_words, features
        word_ft = (mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)
        return word_ft

    def forward(self, inputs, n_words, n_names, qkv):
        mask = self.get_pad_mask(inputs)
        word_ft = self.get_word_ft(inputs, mask, qkv)
        word_ft = word_ft.split(n_words)
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV2'] = WidthCharTransformerMLPHeaderFastPosV2
# WidthCharTransformerMLPHeaderFastPosV2:1 ends here

# WidthCharTransformerMLPHeaderFastPosV3

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV3][WidthCharTransformerMLPHeaderFastPosV3:1]]
class WidthCharTransformerMLPHeaderFastPosV3(WidthCharTransformerMLPHeaderFastPosV2):
    def get_pos(self, max_len):
        freq = self.word_vec_d >> 1
        phrase = np.float32(np.outer(
            range(max_len), 1 / np.power(10000, np.array(range(freq)) / freq)))
        pos = torch.tensor(
            np.hstack([np.cos(phrase), np.sin(phrase)]), device=self.device)
        return pos

    def compute_qkv(self):
        max_len = self.pos.shape[0]
        embedding_w = torch.cat([self.embedding.weight, self.pos])
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        qkv = torch.split(
            self.char_qkv_layer(embedding_w).reshape(qkv_shape),
            [self.n_src_vocab, max_len], dim=0)
        return qkv

    def scaled_dot_product_attention(self, inputs, mask, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = torch.split(char_qkv, [self.n_head] * 3, dim=-2)
        pos_q, pos_k, pos_v = torch.split(pos_qkv, [self.n_head] * 3, dim=-2)
        # n_head, n_src_vocab, n_src_vocab
        char_q = char_q.transpose(0, 1)
        char_k = char_k.transpose(0, 1).transpose(1, 2)
        char_qk = char_q @ char_k
        char_qk[:, :, self.pad_idx] = -1e9
        # n_head, max_len, max_len
        pos_q = pos_q.transpose(0, 1)
        pos_k = pos_k.transpose(0, 1).transpose(1, 2)
        pos_qk = pos_q @ pos_k
        # 1, n_head, n_src_vocab, max_len
        char_q_pos_k = (char_q @ pos_k).unsqueeze(0)
        char_k_pos_q = (pos_q @ char_k).transpose(1, 2).unsqueeze(0)
        # 2, n_head, n_src_vocab, max_len
        char_pos_qk = torch.cat([char_q_pos_k, char_k_pos_q], dim=0)
        # n_words, n_head, max_len, max_len
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # n_words, n_head, max_len, max_len
        inputs_char_pos_qk = char_pos_qk.index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, self.pos.shape[0]]).sum(0).transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk + pos_qk) / self.sdpa_temperature
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, max_len, n_head, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV3'] = WidthCharTransformerMLPHeaderFastPosV3
# WidthCharTransformerMLPHeaderFastPosV3:1 ends here

# WidthCharTransformerMLPHeaderFastPosV4


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV4][WidthCharTransformerMLPHeaderFastPosV4:1]]
class WidthCharTransformerMLPHeaderFastPosV4(WidthCharTransformerMLPHeaderFastPosV2):
    """put pos encoder before linear layer"""
    def get_pos(self, max_chars):
        freq = self.word_vec_d >> 1
        phrase = np.float32(np.outer(
            range(max_chars), 1 / np.power(10000, np.array(range(freq)) / freq)))
        pos = torch.tensor(
            np.hstack([np.cos(phrase), np.sin(phrase)]), device=self.device)
        return pos

    def compute_qkv(self):
        max_chars = self.pos.shape[0]
        embedding_w = torch.cat([self.embedding.weight, self.pos])
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        qkv = torch.split(
            self.bn_layer(self.dropout_layer(self.char_qkv_layer(embedding_w).reshape(qkv_shape))),
            [self.n_src_vocab, max_chars], dim=0)
        return self.interactive_char_pos(qkv)

    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = torch.split(char_qkv, [self.n_head] * 3, dim=-2)
        pos_q, pos_k, pos_v = torch.split(pos_qkv, [self.n_head] * 3, dim=-2)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # n_head, n_src_vocab, n_src_vocab
        char_qk = char_q @ char_k
        char_qk[:, :, self.pad_idx] = -1e9
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # n_head, max_chars, max_chars
        pos_qk = pos_q @ pos_k
        # 1, n_head, n_src_vocab, max_chars
        char_q_pos_k = (char_q @ pos_k).unsqueeze(0)
        char_k_pos_q = (pos_q @ char_k).transpose(1, 2).unsqueeze(0)
        # 2, n_head, n_src_vocab, max_chars
        char_pos_qk = torch.cat([char_q_pos_k, char_k_pos_q], dim=0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, mask, qkv):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # (2, n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_pos_qk = char_pos_qk.index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, self.pos.shape[0]]).sum(0).transpose(0, 1)
        # n_words, n_head, max_chars, max_chars
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk + pos_qk) / self.sdpa_temperature
        # n_words, n_head, max_chars, max_chars
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, max_chars, n_head, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV4'] = WidthCharTransformerMLPHeaderFastPosV4
# WidthCharTransformerMLPHeaderFastPosV4:1 ends here

# WidthCharTransformerMLPHeaderFastPosV5


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV5][WidthCharTransformerMLPHeaderFastPosV5:1]]
class WidthCharTransformerMLPHeaderFastPosV5(WidthCharTransformerMLPHeaderFastPosV4):
    """pos has a separate qkv layer."""
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__(n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda")
        self.pos_qkv_layer = nn.Linear(
            word_vec_d, 3 * n_head * word_vec_d, bias=False)
        self.pos_bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        self.pos_dropout_layer = nn.Dropout(dropout)

    def compute_qkv(self):
        max_len = self.pos.shape[0]
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        char_qkv = self.bn_layer(self.dropout_layer(
            self.char_qkv_layer(embedding_w).reshape(qkv_shape)))
        pos_qkv_shape = [self.pos.shape[0], 3 * self.n_head, self.pos.shape[1]]
        pos_qkv = self.pos_bn_layer(self.pos_dropout_layer(
            self.pos_qkv_layer(self.pos).reshape(pos_qkv_shape)))
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def compute_qkv_no_dropout(self):
        max_len = self.pos.shape[0]
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        char_qkv = self.bn_layer(self.char_qkv_layer(embedding_w).reshape(qkv_shape))
        pos_qkv_shape = [self.pos.shape[0], 3 * self.n_head, self.pos.shape[1]]
        pos_qkv = self.pos_bn_layer(
            self.pos_qkv_layer(self.pos).reshape(pos_qkv_shape))
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def compute_qkv_no_bn(self):
        max_len = self.pos.shape[0]
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        char_qkv = self.dropout_layer(self.char_qkv_layer(embedding_w).reshape(qkv_shape))
        pos_qkv_shape = [self.pos.shape[0], 3 * self.n_head, self.pos.shape[1]]
        pos_qkv = self.pos_dropout_layer(
            self.pos_qkv_layer(self.pos).reshape(pos_qkv_shape))
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def compute_qkv_no_bn_no_dropout(self):
        max_len = self.pos.shape[0]
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        char_qkv = self.char_qkv_layer(embedding_w).reshape(qkv_shape)
        pos_qkv_shape = [self.pos.shape[0], 3 * self.n_head, self.pos.shape[1]]
        pos_qkv = self.pos_qkv_layer(self.pos).reshape(pos_qkv_shape)
        return self.interactive_char_pos([char_qkv, pos_qkv])

model_set['WidthCharTransformerMLPHeaderFastPosV5'] = WidthCharTransformerMLPHeaderFastPosV5
# WidthCharTransformerMLPHeaderFastPosV5:1 ends here

# WidthCharTransformerMLPHeaderFastPosV6


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV6][WidthCharTransformerMLPHeaderFastPosV6:1]]
class WidthCharTransformerMLPHeaderFastPosV6(WidthCharTransformerMLPHeaderFastPosV5):
    """Compute word_ft before mlp.
This means that get n_head word features and use the mlp layer to extract word features."""
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__(n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda")

    def forward(self, inputs, n_words, n_names, qkv):
        # mask shape: n_words, 1, 1, max_chars
        mask = (inputs != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # char_ft shape: n_words, n_head, max_chars, word_vec_d
        char_ft = self.scaled_dot_product_attention(inputs, mask, qkv).transpose(-2, -3)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = ((mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.reshape(list(word_ft.shape[:-2]) + [-1])
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.split(n_words, dim=0)
        # name_ft shape: n_names, vec_d
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV6'] = WidthCharTransformerMLPHeaderFastPosV6
# WidthCharTransformerMLPHeaderFastPosV6:1 ends here

# WidthCharTransformerMLPHeaderFastPosV6_


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV6_][WidthCharTransformerMLPHeaderFastPosV6_:1]]
class WidthCharTransformerMLPHeaderFastPosV6_(WidthCharTransformerMLPHeaderFastPosV5):
    """no bn and dropout at qkv"""
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__(n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda")
        self.pos_qkv_layer = nn.Linear(
            word_vec_d, 3 * n_head * word_vec_d, bias=False)
        delattr(self, "dropout_layer")
        delattr(self, "bn_layer")

    def compute_qkv(self):
        max_len = self.pos.shape[0]
        embedding_w = self.embedding.weight
        qkv_shape = [embedding_w.shape[0], 3 * self.n_head, embedding_w.shape[1]]
        char_qkv = self.char_qkv_layer(embedding_w).reshape(qkv_shape)
        pos_qkv_shape = [self.pos.shape[0], 3 * self.n_head, self.pos.shape[1]]
        pos_qkv = self.pos_qkv_layer(self.pos).reshape(pos_qkv_shape)
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def forward(self, inputs, n_words, n_names, qkv):
        # mask shape: n_words, 1, 1, max_chars
        mask = (inputs != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # char_ft shape: n_words, n_head, max_chars, word_vec_d
        char_ft = self.scaled_dot_product_attention(inputs, mask, qkv).transpose(-2, -3)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = ((mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.reshape(list(word_ft.shape[:-2]) + [-1])
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.split(n_words, dim=0)
        # name_ft shape: n_names, vec_d
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV6_'] = WidthCharTransformerMLPHeaderFastPosV6_
# WidthCharTransformerMLPHeaderFastPosV6_:1 ends here

# WidthCharTransformerMLPHeaderFastPosV7


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*WidthCharTransformerMLPHeaderFastPosV7][WidthCharTransformerMLPHeaderFastPosV7:1]]
class WidthCharTransformerMLPHeaderFastPosV7(WidthCharTransformerMLPHeaderFastPosV5):
    """Add position embeddings for word features by sin/cos functools by before mlp layer."""
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__(n_src_vocab, max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda")
        self.n_src_vocab = int(np.ceil(n_src_vocab / 32) * 32)
        self.max_chars = max_chars
        self.max_chars = int(np.ceil(self.max_chars / 32) * 32)
        self.zeros = torch.tensor([0]*1000, device=device).reshape([1000, 1])
        super().__init__(self.n_src_vocab, self.max_chars, n_head, word_vec_d, dropout, mlp_config, pad_idx=0, device="cuda")
        self.freq = 32
        # self.word_pos shape: 1, n_head, word_vec_d
        self.word_pos = (torch.tensor(list(range(1, int(self.freq / 2) + 1)), dtype=torch.float, device=self.device) / self.freq * 2 * np.pi).unsqueeze(-1).repeat([int(self.n_head / self.freq * 2), self.word_vec_d]).unsqueeze(0)
        mlp_l1_w = self.gen_fourier_weight([mlp_config[0][0], self.n_head * 2])
        self.mlp[0].linear = nn.Linear(self.n_head * 2, mlp_config[0][0])
        self.mlp[0].linear.weight = torch.nn.Parameter(mlp_l1_w)
        self.compute_qkv = self.compute_qkv_no_dropout

    def normal_pdf(self, x, mu=0, sigma=1):
        return 1 / (np.sqrt(np.pi * 2) * sigma) * np.exp(np.power((x - mu), 2) / -2)

    def gen_fourier_weight(self, shape, mu=0, sigma=0.5):
        # w range: (-0.1, 0.1)
        w = np.random.uniform(-0.1, 0.1, shape)
        # p range: (-2, 2)
        p = np.random.uniform(-2, 2, shape)
        return torch.tensor(np.float32(p * self.normal_pdf(p, mu, sigma)))

    def forward(self, inputs, n_words, n_names, qkv):
        # mask shape: n_words, 1, 1, max_chars
        inputs = torch.cat([inputs, self.zeros[:inputs.shape[0]]], dim=-1)
        mask = (inputs != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # char_ft shape: n_words, n_head, max_chars, word_vec_d
        char_ft = self.scaled_dot_product_attention(inputs, mask, qkv).transpose(-2, -3)
        # word_ft shape: n_words, n_head, word_vec_d
        word_ft = ((mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)) * self.word_pos
        word_ft = torch.cat(
            [torch.cos(word_ft), torch.sin(word_ft)], dim=-2).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.reshape(list(word_ft.shape[:-2]) + [-1])
        # word_ft shape: n_words, vec_d
        word_ft = word_ft.split(n_words, dim=0)
        # name_ft shape: n_names, vec_d
        name_ft = torch.stack([i.mean(-2) for i in word_ft])
        output = name_ft.split(n_names)
        return output

model_set['WidthCharTransformerMLPHeaderFastPosV7'] = WidthCharTransformerMLPHeaderFastPosV7
# WidthCharTransformerMLPHeaderFastPosV7:1 ends here

# SVTransformerBPETokenV1


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV1][SVTransformerBPETokenV1:1]]
class SVTransformerBPETokenV1(nn.Module):
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d,
                 dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__()
        # attributes
        self.n_src_vocab = n_src_vocab
        self.max_chars = max_chars
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.dropout = dropout
        self.mlp_config = mlp_config
        self.device = device
        self.pad_idx = pad_idx
        self.mha_dim = n_head
        self.char_emb = nn.Embedding(n_src_vocab, word_vec_d, padding_idx=pad_idx).to(device)
        self.char = self.char_emb.weight
        self.pos = self.get_pos(self.max_chars).to(device)
        self.char_qkv_layer = ComputeQKV(self.n_head, self.word_vec_d)
        self.pos_qkv_layer = ComputeQKV(self.n_head, self.word_vec_d)
        # sdpa: scaled dot product attention
        self.sdpa_temperature = word_vec_d ** 0.5
        self.sdpa_dropout = nn.Dropout(dropout)
        self.sdpa_bn_layer = nn.LayerNorm(word_vec_d, eps=1e-5)
        # layers
        self.mlp_dims = [self.mha_dim] + [c[0] for c in mlp_config[:-1]]
        self.mlp = nn.Sequential(*[
            Dense(i, *c)
            for i, c in zip(self.mlp_dims, mlp_config)])

    def get_pos(self, max_chars):
        freq = self.word_vec_d >> 1
        phrase = np.float32(np.outer(
            range(max_chars), 1 / np.power(10000, np.array(range(freq)) / freq)))
        pos = torch.tensor(
            np.hstack([np.cos(phrase), np.sin(phrase)]), device=self.device)
        return pos

    def compute_qkv(self, max_len):
        char_qkv = self.char_qkv_layer(self.char)
        pos_qkv = self.pos_qkv_layer(self.pos[:max_len])
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [char_qkv[:, i] for i in range(char_qkv.shape[1])]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # n_head, n_src_vocab, n_src_vocab
        char_qk = char_q @ char_k
        char_qk[:, :, self.pad_idx] = -1e9
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # n_head, max_chars, max_chars
        pos_qk = pos_q @ pos_k
        # 1, n_head, n_src_vocab, max_chars
        char_q_pos_k = (char_q @ pos_k).unsqueeze(0)
        char_k_pos_q = (pos_q @ char_k).transpose(1, 2).unsqueeze(0)
        # 2, n_head, n_src_vocab, max_chars
        char_pos_qk = torch.cat([char_q_pos_k, char_k_pos_q], dim=0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, mask, qkv):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # (2, n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_pos_qk = char_pos_qk.index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, pos_qk.shape[-1]]).sum(0).transpose(0, 1)
        # n_words, n_head, max_chars, max_chars
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk + pos_qk) / self.sdpa_temperature
        # n_words, n_head, max_chars, max_chars
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, max_chars, n_head, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def forward(self, char_code, word_code, n_words, n_names, qkv):
        # mask shape: n_words, 1, 1, max_chars
        mask = (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # char_ft shape: n_words, n_head, max_chars, word_vec_d
        char_ft = self.scaled_dot_product_attention(char_code, mask, qkv).transpose(-2, -3)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = ((mask.float() @ char_ft).squeeze(-2) / mask.sum(-1)).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, word_vec_d
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        # word_ft shape: len(word_code), word_vec_d
        word_ft = word_ft.index_select(0, word_code)
        # word_ft shape: n_words, word_vec_d
        word_ft = word_ft.split(n_words, dim=0)
        # name_ft shape: n_names, word_vec_d
        name_ft = torch.stack([i.mean(0) for i in word_ft])
        output = name_ft.split(n_names)
        return output

model_set['SVTransformerBPETokenV1'] = SVTransformerBPETokenV1
# SVTransformerBPETokenV1:1 ends here

# SVTransformerBPETokenV2

# 根据长度进行切片，并自适应长度进行训练


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV2][SVTransformerBPETokenV2:1]]
class SVTransformerBPETokenV2(SVTransformerBPETokenV1):
    def compute_qkv(self):
        char_qkv = self.char_qkv_layer(self.char)
        pos_qkv = self.pos_qkv_layer(self.pos)
        return self.interactive_char_pos([char_qkv, pos_qkv])

    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # (2, n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_pos_qk = char_pos_qk[:, :, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, max_len]).sum(0).transpose(0, 1)
        # n_words, n_head, max_chars, max_chars
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_chars, max_chars
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, max_chars, n_head, word_vec_d
        pos_v = pos_v[:max_len].transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def forward(self, char_code, qkv, max_len):
        # mask shape: n_words, 1, 1, max_chars
        mask = (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # char_ft shape: n_words, n_head, max_chars, word_vec_d
        char_ft = self.scaled_dot_product_attention(
            char_code, mask, qkv, max_len).transpose(-2, -3)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = ((mask.float() @ char_ft).squeeze(-2) /
                   mask.sum(-1)).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, word_vec_d
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        return word_ft

model_set['SVTransformerBPETokenV2'] = SVTransformerBPETokenV2
# SVTransformerBPETokenV2:1 ends here

# SVTransformerBPETokenV3

# 在计算 QKV 时，计算对数，Batch_size 为 32, 64 时可以快 1-2 秒，但有可能报 Loss 为 nan 的问题，未解决


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV3][SVTransformerBPETokenV3:1]]
class SVTransformerBPETokenV3(SVTransformerBPETokenV2):
    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [char_qkv[:, i] for i in range(char_qkv.shape[1])]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # n_head, n_src_vocab, n_src_vocab
        char_qk = char_q @ char_k
        char_qk[:, :, self.pad_idx] = -1e9
        char_qk = torch.exp(char_qk / self.sdpa_temperature)
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # n_head, max_chars, max_chars
        pos_qk = torch.exp(pos_q @ pos_k / self.sdpa_temperature)
        # 1, n_head, n_src_vocab, max_chars
        char_q_pos_k = (char_q @ pos_k).unsqueeze(0)
        char_k_pos_q = (pos_q @ char_k).transpose(1, 2).unsqueeze(0)
        # 2, n_head, n_src_vocab, max_chars
        char_pos_qk = torch.exp(torch.cat([char_q_pos_k, char_k_pos_q], dim=0) / self.sdpa_temperature)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # (2, n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_pos_qk = char_pos_qk[:, :, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, max_len]).sum(0).transpose(0, 1)
        # n_words, n_head, max_chars, max_chars
        inputs_attn = inputs_char_qk * inputs_char_pos_qk * pos_qk[:, :max_len, :max_len]
        inputs_attn = self.sdpa_dropout(inputs_attn / inputs_attn.sum(-1).unsqueeze(-1))
        # n_words, max_chars, n_head, word_vec_d
        pos_v = pos_v[:max_len].transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

model_set['SVTransformerBPETokenV3'] = SVTransformerBPETokenV3
# SVTransformerBPETokenV3:1 ends here

# SVTransformerBPETokenV4


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV4][SVTransformerBPETokenV4:1]]
class SVTransformerBPETokenV4(SVTransformerBPETokenV2):
    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(1, i).index_select(-1, i).unsqueeze(0)
             for i in inputs])
        # (2, n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_chars, max_chars)
        inputs_char_pos_qk = char_pos_qk[:, :, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [2, self.n_head, *inputs.shape, max_len]).sum(0).transpose(0, 1)
        # n_words, n_head, max_chars, max_chars
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_chars, max_chars
        inputs_attn = nn.functional.softmax(inputs_sim, dim=-1)
        # n_words, max_chars, n_head, word_vec_d
        pos_v = pos_v[:max_len].transpose(0, 1).unsqueeze(0)
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2).add(pos_v)
        # output shape: n_words, max_chars, n_head, word_vec_d
        output = (inputs_attn @ inputs_v).transpose(1, 2).contiguous()
        # inputs_emb = self.char_emb(inputs)
        # output = self.sdpa_bn_layer(self.sdpa_dropout(output) + inputs_emb.unsqueeze(-2))
        output = self.sdpa_bn_layer(self.sdpa_dropout(output))
        return output

model_set['SVTransformerBPETokenV4'] = SVTransformerBPETokenV4
# SVTransformerBPETokenV4:1 ends here

# SVTransformerBPETokenV5

# 优化 interactive_char_pos 的计算。


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV5][SVTransformerBPETokenV5:1]]
class SVTransformerBPETokenV5(SVTransformerBPETokenV2):
    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [char_qkv[:, i] for i in range(char_qkv.shape[1])]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # 1, n_head, n_src_vocab, n_src_vocab
        char_qk = (char_q @ char_k).unsqueeze(0)
        # char_qk[:, :, :, self.pad_idx] = -1e9
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_chars, n_head, word_vec_d) -> (1, n_head, max_chars, word_vec_d)
        pos_qk = (pos_q @ pos_k).unsqueeze(0)
        # n_head, n_src_vocab, max_chars
        char_pos_qk = (char_q @ pos_k) + (pos_q @ char_k).transpose(1, 2)
        # max_chars, n_head, word_vec_d -> 1, n_head, max_chars, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (1, n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(-2, i).index_select(-1, i)
             for i in inputs])
        # (n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [self.n_head, *inputs.shape, max_len]).transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, n_head, max_len
        inputs_attn = (mask * inputs_attn).sum(-2)
        # 1, n_head, max_len, word_vec_d
        pos_v = pos_v[:, :, :max_len]
        # max_chars, n_head, word_vec_d -> n_words, n_head, max_len, word_vec_d
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2)
        # output shape: n_words, n_head, word_vec_d
        output = torch.einsum("whl, whld->whd", inputs_attn, (pos_v + inputs_v))
        # # output shape: n_words, max_chars, n_head, word_vec_d
        # output = (inputs_attn @ (pos_v + inputs_v)).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def forward(self, char_code, qkv, max_len):
        # mask shape: n_words, 1, 1, max_chars
        mask = (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.scaled_dot_product_attention(
            char_code, mask, qkv, max_len).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, word_vec_d
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        return word_ft

model_set['SVTransformerBPETokenV5'] = SVTransformerBPETokenV5
# SVTransformerBPETokenV5:1 ends here

# SVTransformerBPETokenV6

# 采用全局 [CLS] 来生成注意力权重


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV6][SVTransformerBPETokenV6:1]]
class SVTransformerBPETokenV6(SVTransformerBPETokenV5):
    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (1, n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(-2, i).index_select(-1, i)
             for i in inputs])
        # (n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [self.n_head, *inputs.shape, max_len]).transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, n_head, max_len
        inputs_attn = (mask * inputs_attn)[:, :, 0]
        # 1, n_head, max_len, word_vec_d
        pos_v = pos_v[:, :, :max_len]
        # max_chars, n_head, word_vec_d -> n_words, n_head, max_len, word_vec_d
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2)
        # output shape: n_words, n_head, word_vec_d
        output = torch.einsum("whl, whld->whd", inputs_attn, (pos_v + inputs_v))
        # # output shape: n_words, max_chars, n_head, word_vec_d
        # output = (inputs_attn @ (pos_v + inputs_v)).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

model_set['SVTransformerBPETokenV6'] = SVTransformerBPETokenV6
# SVTransformerBPETokenV6:1 ends here

# SVTransformerBPETokenV7

# 优化 interactive_char_pos 的计算。


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV7][SVTransformerBPETokenV7:1]]
class SVTransformerBPETokenV7(SVTransformerBPETokenV2):
    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [char_qkv[:, i] for i in range(char_qkv.shape[1])]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # 1, n_head, n_src_vocab, n_src_vocab
        char_qk = (char_q @ char_k).unsqueeze(0)
        char_qk[:, :, :, self.pad_idx] = -1e9
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_chars, n_head, word_vec_d) -> (1, n_head, max_chars, word_vec_d)
        pos_qk = (pos_q @ pos_k).unsqueeze(0)
        # n_head, n_src_vocab, max_chars
        char_pos_qk = (char_q @ pos_k) + (pos_q @ char_k).transpose(1, 2)
        # max_chars, n_head, word_vec_d -> 1, n_head, max_chars, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, char_v, pos_v

    def scaled_dot_product_attention(self, inputs, mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, char_v, pos_v = qkv
        # (1, n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(-2, i).index_select(-1, i)
             for i in inputs])
        # (n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [self.n_head, *inputs.shape, max_len]).transpose(0, 1)
        # n_words, n_head, max_len, max_len
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, n_head, max_len
        inputs_attn = (mask.float() @ inputs_attn).squeeze(-2) / mask.sum(-1)
        # 1, n_head, max_len, word_vec_d
        pos_v = pos_v[:, :, :max_len]
        # max_chars, n_head, word_vec_d -> n_words, n_head, max_len, word_vec_d
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2)
        # output shape: n_words, n_head, word_vec_d
        output = torch.einsum("whl, whld->whd", inputs_attn, (pos_v + inputs_v))
        # # output shape: n_words, max_chars, n_head, word_vec_d
        # output = (inputs_attn @ (pos_v + inputs_v)).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

    def forward(self, char_code, qkv, max_len):
        # mask: n_words, 1, 1, max_chars
        mask = (char_code != self.pad_idx).unsqueeze(-2).unsqueeze(-2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.scaled_dot_product_attention(
            char_code, mask, qkv, max_len).transpose(-1, -2)
        # word_ft shape: n_words, word_vec_d, n_head
        word_ft = self.mlp(word_ft)
        # word_ft shape: n_words, word_vec_d
        word_ft = word_ft.reshape([word_ft.shape[0], -1])
        return word_ft

model_set['SVTransformerBPETokenV7'] = SVTransformerBPETokenV7
# SVTransformerBPETokenV7:1 ends here

# SVTransformerBPETokenV8

# 分开 char_pos_qk 和 pos_char_qk


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV8][SVTransformerBPETokenV8:1]]
class SVTransformerBPETokenV8(SVTransformerBPETokenV7):
    def interactive_char_pos(self, qkv):
        char_qkv, pos_qkv = qkv
        char_q, char_k, char_v = [char_qkv[:, i] for i in range(char_qkv.shape[1])]
        pos_q, pos_k, pos_v = [pos_qkv[:, i] for i in range(pos_qkv.shape[1])]
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, n_src_vocab, word_vec_d)
        char_q = char_q.transpose(0, 1)
        # (n_src_vocab, n_head, word_vec_d) -> (n_head, word_vec_d, n_src_vocab)
        char_k = char_k.permute(1, -1, 0)
        # 1, n_head, n_src_vocab, n_src_vocab
        char_qk = (char_q @ char_k).unsqueeze(0)
        char_qk[:, :, :, self.pad_idx] = -1e9
        # (max_chars, n_head, word_vec_d) -> (n_head, max_chars, word_vec_d)
        pos_q = pos_q.transpose(0, 1)
        # (max_chars, n_head, word_vec_d) -> (n_head, word_vec_d, max_chars)
        pos_k = pos_k.permute(1, -1, 0)
        # (max_chars, n_head, word_vec_d) -> (1, n_head, max_chars, word_vec_d)
        pos_qk = (pos_q @ pos_k).unsqueeze(0)
        # n_head, n_src_vocab, max_chars
        char_pos_qk = (char_q @ pos_k)
        # n_head, max_chars, n_src_vocab
        pos_char_qk = (pos_q @ char_k)
        # max_chars, n_head, word_vec_d -> 1, n_head, max_chars, word_vec_d
        pos_v = pos_v.transpose(0, 1).unsqueeze(0)
        return char_qk, pos_qk, char_pos_qk, pos_char_qk, char_v, pos_v
    def scaled_dot_product_attention(self, inputs, not_mask, qkv, max_len):
        char_qk, pos_qk, char_pos_qk, pos_char_qk, char_v, pos_v = qkv
        # (1, n_head, n_src_vocab, n_src_vocab) -> (n_words, n_head, max_len, max_len)
        inputs_char_qk = torch.cat(
            [char_qk.index_select(-2, i).index_select(-1, i)
             for i in inputs])
        # (n_head, n_src_vocab, max_chars) -> (n_words, n_head, max_len, max_len)
        inputs_char_pos_qk = char_pos_qk[:, :, :max_len].index_select(
            -2, inputs.reshape(-1)).reshape(
                [self.n_head, *inputs.shape, max_len]).transpose(0, 1)
        # (n_head, max_chars, n_src_vocab) -> (n_head, max_len, n_words, max_len) -> (n_words, n_head, max_len, max_len)
        inputs_pos_char_qk = pos_char_qk[:, :max_len].index_select(
            -1, inputs.reshape(-1)).reshape(
                [self.n_head, max_len, *inputs.shape]).permute(-2, 0, 1, -1)
        inputs_char_pos_qk = inputs_char_pos_qk + inputs_pos_char_qk
        # n_words, n_head, max_len, max_len
        inputs_sim = (inputs_char_qk + inputs_char_pos_qk +
                      pos_qk[:, :, :max_len, :max_len]) / self.sdpa_temperature
        # n_words, n_head, max_len, max_len
        inputs_attn = self.sdpa_dropout(
            nn.functional.softmax(inputs_sim, dim=-1))
        # n_words, n_head, max_len
        inputs_attn = (not_mask.float() @ inputs_attn).squeeze(-2) / not_mask.sum(-1)
        # 1, n_head, max_len, word_vec_d
        pos_v = pos_v[:, :, :max_len]
        # max_chars, n_head, word_vec_d -> n_words, n_head, max_len, word_vec_d
        inputs_v = char_v.index_select(0, inputs.reshape(-1)).reshape(
            [*inputs.shape, self.n_head, self.word_vec_d]).transpose(1, 2)
        # output shape: n_words, n_head, word_vec_d
        output = torch.einsum("whl, whld->whd", inputs_attn, (pos_v + inputs_v))
        # # output shape: n_words, max_chars, n_head, word_vec_d
        # output = (inputs_attn @ (pos_v + inputs_v)).transpose(1, 2).contiguous()
        output = self.sdpa_bn_layer(output)
        return output

model_set['SVTransformerBPETokenV8'] = SVTransformerBPETokenV8
# SVTransformerBPETokenV8:1 ends here

# SVTransformerBPETokenV7CE

# 添加线性变换层以配适交叉熵损失


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV7CE][SVTransformerBPETokenV7CE:1]]
class SVTransformerBPETokenV7CE(SVTransformerBPETokenV7):
      def __init__(self, n_classes, n_src_vocab, max_chars, n_head, word_vec_d,
                   dropout, mlp_config, pad_idx=0, device="cuda"):
          super().__init__(n_src_vocab, max_chars, n_head, word_vec_d,
                           dropout, mlp_config, pad_idx=0, device="cuda")
          self.n_classes = n_classes
          self.output_layer = nn.Linear(self.word_vec_d * self.mlp_config[-1][0],
                                        n_classes)

model_set['SVTransformerBPETokenV7CE'] = SVTransformerBPETokenV7CE
# SVTransformerBPETokenV7CE:1 ends here

# SVTransformerBPETokenV7CE

# 添加线性变换层以配适交叉熵损失


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenV7CE][SVTransformerBPETokenV7CE:1]]
class SVTransformerBPETokenV8CE(SVTransformerBPETokenV8):
      def __init__(self, n_classes, n_src_vocab, max_chars, n_head, word_vec_d,
                   dropout, mlp_config, pad_idx=0, device="cuda"):
          super().__init__(n_src_vocab, max_chars, n_head, word_vec_d,
                           dropout, mlp_config, pad_idx=0, device="cuda")
          self.n_classes = n_classes
          self.output_layer = nn.Linear(self.word_vec_d * self.mlp_config[-1][0],
                                        n_classes)

model_set['SVTransformerBPETokenV8CE'] = SVTransformerBPETokenV8CE
# SVTransformerBPETokenV7CE:1 ends here

# SVTransformerBPETokenWordPadV1

# 优化 interactive_char_pos 的计算。


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/model.org::*SVTransformerBPETokenWordPadV1][SVTransformerBPETokenWordPadV1:1]]
class SVTransformerBPETokenWordPadV1(SVTransformerBPETokenV5):
    def __init__(self, n_src_vocab, max_chars, n_head, word_vec_d,
                 dropout, mlp_config, pad_idx=0, device="cuda"):
        super().__init__(n_src_vocab, max_chars, n_head, word_vec_d,
                 dropout, mlp_config, pad_idx=0, device="cuda")
        self.mention_dropout = nn.Dropout(dropout)

model_set['SVTransformerBPETokenWordPadV1'] = SVTransformerBPETokenWordPadV1
# SVTransformerBPETokenWordPadV1:1 ends here
