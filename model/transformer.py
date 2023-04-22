import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 hidden_layers: int,
                 hidden_heads: int,
                 hidden_feedforward_dim: int,
                 hidden_dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.hidden_heads = hidden_heads
        self.hidden_feedforward_dim = hidden_feedforward_dim
        self.hidden_dropout = hidden_dropout

        transform_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=hidden_heads,
                                                   dim_feedforward=hidden_feedforward_dim,
                                                   dropout=hidden_dropout,
                                                   activation="gelu",
                                                   batch_first=True)
        layer_norm = nn.LayerNorm(hidden_dim)
        self.transform = nn.TransformerEncoder(encoder_layer=transform_layer,
                                               num_layers=hidden_layers,
                                               norm=layer_norm,
                                               enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        return self.transform(x, src_key_padding_mask=padding_mask)



