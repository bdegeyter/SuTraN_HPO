import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot product attention for the num_heads heads. 

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        mask : torch.Tensor, optional
            Padding mask, by default None. 
            If not None, shape (batch_size, window_size) and of the 
            bool dtype, with True on the positions that correspond to 
            padded / masked events. 

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        # Flash Attention path (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            # F.scaled_dot_product_attention bool mask: True = attend.
            # Our convention: True = padded (don't attend) -> negate.
            attn_mask = ~mask.unsqueeze(1).unsqueeze(1) if mask is not None else None
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        # Fallback: manual implementation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, H, T_q, T_k)
        if mask is not None: # (B, T_k)
            # (B, T_k) → (B, 1, 1, T_k), broadcasts to (B, H, T_q, T_k)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), value=-1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None, enc_kv_cache=None):
        """MHA

        Parameters
        ----------
        Q : torch.Tensor 
            Queries. Tensor of shape (batch_size, T_q, d_model).
        K : torch.Tensor 
            Keys. Tensor of shape (batch_size, T_k, d_model).
            Ignored when enc_kv_cache is provided.
        V : torch.Tensor 
            Values. Tensor of shape (batch_size, T_k, d_model).
            Ignored when enc_kv_cache is provided.
        mask : torch.Tensor, optional
            Boolean mask of shape (batch_size, T_k). Entries are 
            True for the embeddings that correspond to padded events. 
        enc_kv_cache : tuple of torch.Tensor or None, optional
            Pre-projected (K, V) each of shape (B, H, T_k, d_k).
            When provided, skips K and V projection.

        Returns
        -------
        output : torch.Tensor
            Shape (batch_size, T_q, d_model).
        kv : tuple of torch.Tensor
            (K_proj, V_proj) each of shape (B, H, T_k, d_k).
        """
        Q = self.split_heads(self.W_q(Q)) # (B, H, T_q, d_k)
        if enc_kv_cache is not None:
            K, V = enc_kv_cache
        else:
            K = self.split_heads(self.W_k(K)) # (B, H, T_k, d_k)
            V = self.split_heads(self.W_v(V)) # (B, H, T_k, d_k)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, (K, V)

# Self attention for in decoder. Seperate class because of fixed look-ahead mask. 
class MultiHeadSelfAttentionDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttentionDecoder, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot product attention for the num_heads heads. 

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        # Flash Attention path with causal mask (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)

        # Fallback: manual implementation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, num_heads, W, W)
        window_size = Q.shape[2]
        look_ahead = torch.triu(torch.ones(1, 1, window_size, window_size), diagonal=1).bool().to(device)
        attn_scores = attn_scores.masked_fill(mask=look_ahead, value=-1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, kv_cache=None):
        """MHA with optional KV caching for autoregressive inference.

        Parameters
        ----------
        Q, K, V : torch.Tensor
            Shape (B, W, d_model) during training; (B, 1, d_model)
            for a single new token during inference.
        kv_cache : tuple of torch.Tensor or None
            (K_cached, V_cached) from previous decoding steps. None
            during training and at the first inference step.

        Returns
        -------
        output : torch.Tensor
            Shape (B, T_q, d_model).
        new_kv_cache : tuple of torch.Tensor or None
            Updated KV cache. None during training.
        """
        Q_proj = self.split_heads(self.W_q(Q))
        K_proj = self.split_heads(self.W_k(K))
        V_proj = self.split_heads(self.W_v(V))

        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K_proj = torch.cat([K_cached, K_proj], dim=2)
            V_proj = torch.cat([V_cached, V_proj], dim=2)

        if self.training:
            # Full sequence with causal look-ahead mask
            attn_output = self.scaled_dot_product_attention(Q_proj, K_proj, V_proj)
            return self.W_o(self.combine_heads(attn_output)), None
        else:
            # Single-token query: no causal mask needed since Q only
            # attends to past tokens already in the cache
            attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, V_proj)
            return self.W_o(self.combine_heads(attn_output)), (K_proj, V_proj)
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) # (batch_size, window_size, d_model)