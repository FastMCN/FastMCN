import torch
import numpy as np
from utils.utils import *
from utils import activate
from torch import nn
from torch.nn.functional import normalize

model_set = {}

class ComputeQKV(nn.Module):
    def __init__(self, n_head, word_vec_d, n_layer=3, bias=False):
        super().__init__()
        self.n_head = n_head
        self.word_vec_d = word_vec_d
        self.n_layer = n_layer
        self.bias = bias
        out_features = self.n_layer * self.n_head * self.word_vec_d
        self.linear = nn.Linear(word_vec_d, out_features, bias=self.bias)

    def forward(self, inputs):
        qkv_shape = [inputs.shape[0], self.n_layer, self.n_head, -1]
        return self.linear(inputs).reshape(qkv_shape)

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

class TE(nn.Module):
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

model_set['TE'] = TE

class IE(nn.Module):
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

    def compute_qkv(self):
        char_qkv = self.char_qkv_layer(self.char)
        pos_qkv = self.pos_qkv_layer(self.pos)
        return self.interactive_char_pos([char_qkv, pos_qkv])

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

model_set['IE'] = IE

class IECE(IE):
      def __init__(self, n_classes, n_src_vocab, max_chars, n_head, word_vec_d,
                   dropout, mlp_config, pad_idx=0, device="cuda"):
          super().__init__(n_src_vocab, max_chars, n_head, word_vec_d,
                           dropout, mlp_config, pad_idx=0, device="cuda")
          self.n_classes = n_classes
          self.output_layer = nn.Linear(self.word_vec_d * self.mlp_config[-1][0],
                                        n_classes)

model_set['IECE'] = IECE
