from torch import nn

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
