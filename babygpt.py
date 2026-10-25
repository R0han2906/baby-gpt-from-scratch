"""
BabyGPT — Decoder-only Transformer from Scratch
Architecture: LLaMA 3 / Mistral style (2024)

Components:
  - RoPE (Rotary Position Embedding)
  - RMSNorm (Root Mean Square Normalization)
  - SwiGLU (Gated Feed-Forward Network)
  - GQA (Grouped-Query Attention)
  - SWA (Sliding Window Attention)
  - Weight Tying
  - KV Cache (with causal prefill)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (Zhang & Sennrich, 2019)"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.gamma


class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)"""
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        angles = torch.outer(torch.arange(max_seq_len).float(), freq)
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        seq_len = x.shape[1]
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).view(x.shape)


class TokenEmbedding(nn.Module):
    """Token Embedding + RMSNorm (no positional encoding — RoPE handles it)"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.rmsN = RMSNorm(d_model)

    def forward(self, tokens):
        return self.rmsN(self.embedding_layer(tokens))


class CausalSelfAttention(nn.Module):
    """
    Grouped-Query Attention with Sliding Window + KV Cache
    (Ainslie et al., 2023 | Beltagy et al., 2020)
    """
    def __init__(self, d_model, num_heads, num_kv_heads=2, window_size=128, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0 and num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.window_size = window_size

        self.W_Q = nn.Linear(d_model, d_model)
        kv_dim = num_kv_heads * self.d_k
        self.W_K = nn.Linear(d_model, kv_dim)
        self.W_V = nn.Linear(d_model, kv_dim)
        self.W_O = nn.Linear(d_model, d_model)

        self.rope = RoPE(d_model)
        self.rope_kv = RoPE(kv_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Training forward: full sequence with causal + window mask."""
        B, T, _ = x.shape

        Q = self.rope(self.W_Q(x), start_pos=0)
        K = self.rope_kv(self.W_K(x), start_pos=0)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal + Sliding Window mask
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        window = torch.tril(torch.full((T, T), float('-inf'), device=x.device), diagonal=-self.window_size)
        scores = scores + torch.minimum(causal, window)

        w = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(w, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_dropout(self.W_O(out))

    def forward_with_cache(self, x, cache=None, seq_pos=0):
        """Inference forward: KV cache with causal prefill mask."""
        B, T, _ = x.shape

        Q = self.rope(self.W_Q(x), start_pos=seq_pos)
        K = self.rope_kv(self.W_K(x), start_pos=seq_pos)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            K = torch.cat([cache[0], K], dim=2)
            V = torch.cat([cache[1], V], dim=2)

        # Sliding window trim
        if K.shape[2] > self.window_size:
            K = K[:, :, -self.window_size:]
            V = V[:, :, -self.window_size:]

        new_cache = (K, V)

        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask during prefill (multi-token input)
        if T > 1:
            past = K.shape[2] - T
            mask = torch.zeros(T, K.shape[2], device=x.device)
            mask[:, past:] = torch.triu(
                torch.full((T, T), float('-inf'), device=x.device), diagonal=1
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        w = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), new_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden)
        self.w_gate = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        signal = self.w1(x) * torch.sigmoid(self.w1(x))  # Swish
        return self.dropout(self.w2(signal * self.w_gate(x)))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block (LLaMA style)"""
    def __init__(self, d_model, num_heads, num_kv_heads=2, window_size=128, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, window_size, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_with_cache(self, x, cache=None, seq_pos=0):
        out, new_cache = self.attn.forward_with_cache(self.norm1(x), cache, seq_pos)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class BabyGPT(nn.Module):
    """
    Decoder-only Transformer Language Model

    Architecture: Embed → [Block × N] → RMSNorm → Linear → Logits
    """
    def __init__(self, vocab_size, d_model, num_heads, n_layers,
                 num_kv_heads=2, window_size=128, dropout=0.1):
        super().__init__()
        self.embeddings = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_kv_heads, window_size, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (Press & Wolf, 2017)
        self.head.weight = self.embeddings.embedding_layer.weight

    def forward(self, tokens):
        x = self.embeddings(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))