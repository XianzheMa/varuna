import torch.nn as nn
from .gpt_modules import *
from varuna import CutPoint


class VarunaGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_heads, num_layers):
        super().__init__()
        layers = []

        layers.append(GPTEmbedding(vocab_size, embedding_dim, seq_length))
        for _ in range(num_layers):
            layers.append(GPTTransformerLayer(embedding_dim, num_heads, embedding_dim * 4))
        layers.append(GPTLMHead(embedding_dim, vocab_size))

        self.layers = nn.ModuleList(layers)
        self.cutpoints = nn.ModuleList([CutPoint() for _ in range(len(self.layers) - 1)])

        self.loss_fn = gpt_loss_func

    def forward(self, input_ids, targets):
        x = input_ids
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.cutpoints[i](x)
        return self.loss_fn(x, targets)
