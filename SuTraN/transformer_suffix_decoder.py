import torch
import torch.nn as nn
from SuTraN.layers import MultiHeadAttention, PositionWiseFeedForward, MultiHeadSelfAttentionDecoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadSelfAttentionDecoder(d_model, num_heads)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, cache=None):
        """Forward pass with optional KV caching for inference.

        Parameters
        ----------
        x : torch.Tensor
            (B, W, d_model) during training; (B, 1, d_model) during
            cached inference.
        enc_output : torch.Tensor
            Encoder output, (B, W_enc, d_model).
        src_mask : torch.Tensor
            Encoder padding mask, (B, W_enc).
        cache : tuple or None
            (sa_cache, ca_cache) from previous decoding steps.

        Returns
        -------
        x : torch.Tensor
            Updated hidden state, same shape as input x.
        new_cache : tuple
            (new_sa_cache, new_ca_cache) for the next decoding step.
        """
        sa_cache, ca_cache = cache if cache is not None else (None, None)

        attn_output, new_sa_cache = self.self_attn(x, x, x, kv_cache=sa_cache)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, new_ca_cache = self.cross_attn(x, enc_output, enc_output, src_mask, enc_kv_cache=ca_cache)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, (new_sa_cache, new_ca_cache)