"""
Generation script for BabyGPT (Slow + Fast KV Cache)
Run: python generate.py
"""

import torch
import torch.nn.functional as F
import time
import os
from babygpt import BabyGPT

BLOCK_SIZE = 128
VOCAB_SIZE = 65
D_MODEL = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
N_LAYERS = 6
WINDOW_SIZE = 128
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_vocab():
    text = open('input.txt', 'r').read()
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return char2idx, idx2char


def encode(s, char2idx):
    return [char2idx[c] for c in s]


def decode(lst, idx2char):
    return ''.join(idx2char[i] for i in lst)


def generate_slow(model, seed, char2idx, idx2char, max_tokens=500, temp=0.8, top_k=40):
    model.eval()
    tokens = torch.tensor([encode(seed, char2idx)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens[:, -BLOCK_SIZE:])
            last = logits[:, -1, :] / max(temp, 1e-6)
            if top_k:
                v, _ = torch.topk(last, top_k)
                last[last < v[:, [-1]]] = -float('inf')
            tokens = torch.cat([tokens, torch.multinomial(F.softmax(last, -1), 1)], dim=1)
    return decode(tokens[0].tolist(), idx2char)


def generate_fast(model, seed, char2idx, idx2char, max_tokens=500, temp=0.8, top_k=40):
    model.eval()
    tokens = torch.tensor([encode(seed, char2idx)], dtype=torch.long, device=DEVICE)
    kv_cache = [None] * len(model.blocks)
    seq_pos = 0

    with torch.no_grad():
        x = model.embeddings(tokens)
        for i, block in enumerate(model.blocks):
            x, kv_cache[i] = block.forward_with_cache(x, kv_cache[i], seq_pos)
        seq_pos += tokens.shape[1]

        for _ in range(max_tokens):
            if seq_pos < BLOCK_SIZE:
                logits = model.head(model.final_norm(x))
            else:
                logits = model(tokens[:, -BLOCK_SIZE:])

            last = logits[:, -1, :] / max(temp, 1e-6)
            if top_k:
                v, _ = torch.topk(last, top_k)
                last[last < v[:, [-1]]] = -float('inf')
            next_tok = torch.multinomial(F.softmax(last, -1), 1)
            tokens = torch.cat([tokens, next_tok], dim=1)

            if seq_pos < BLOCK_SIZE:
                x = model.embeddings(next_tok)
                for i, block in enumerate(model.blocks):
                    x, kv_cache[i] = block.forward_with_cache(x, kv_cache[i], seq_pos)
            seq_pos += 1

    return decode(tokens[0].tolist(), idx2char)


def main():
    char2idx, idx2char = load_vocab()
    model = BabyGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, N_LAYERS, NUM_KV_HEADS, WINDOW_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load('babygpt_final.pth', map_location=DEVICE))
    model.eval()

    seed = "ROMEO:"
    print(f"🎭 Seed: {seed}\n")

    t0 = time.time()
    slow = generate_slow(model, seed, char2idx, idx2char, 300)
    t_slow = time.time() - t0

    t0 = time.time()
    fast = generate_fast(model, seed, char2idx, idx2char, 300)
    t_fast = time.time() - t0

    print(f"🐢 Slow: {t_slow:.2f}s | ⚡ Fast: {t_fast:.2f}s | 🚀 {t_slow/t_fast:.1f}x\n")
    print("=" * 60)
    print("SLOW (ground truth):")
    print(slow)
    print("\nFAST (KV cache):")
    print(fast)
    print("=" * 60)


if __name__ == '__main__':
    main()