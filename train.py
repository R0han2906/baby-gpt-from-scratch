"""
Training script for BabyGPT
Run: python train.py
"""

import torch
import torch.nn as nn
import math
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
from babygpt import BabyGPT


# ── Config ──
BLOCK_SIZE = 128
BATCH_SIZE = 64
VOCAB_SIZE = 65
D_MODEL = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
N_LAYERS = 6
WINDOW_SIZE = 128
DROPOUT = 0.1
TOTAL_STEPS = 8000
WARMUP_STEPS = 300
MAX_LR = 5e-4
MIN_LR = 5e-5
EVAL_INTERVAL = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Data ──
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data, self.block_size = data, block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]


def load_data():
    if not os.path.exists('input.txt'):
        print("📥 Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            'input.txt'
        )
    text = open('input.txt', 'r').read()
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars)


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (
        1 + math.cos(math.pi * (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS))
    )


@torch.no_grad()
def estimate_val_loss(model, val_loader, loss_fn, num_batches=20):
    model.eval()
    losses, vi = [], iter(val_loader)
    for _ in range(num_batches):
        try:
            bx, by = next(vi)
        except StopIteration:
            break
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        losses.append(loss_fn(model(bx).reshape(-1, VOCAB_SIZE), by.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    train_data, val_data, vocab_size = load_data()
    train_loader = DataLoader(CharDataset(train_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(CharDataset(val_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = BabyGPT(vocab_size, D_MODEL, NUM_HEADS, N_LAYERS, NUM_KV_HEADS, WINDOW_SIZE, DROPOUT).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parameters: {params:,}")
    print(f"📱 Device: {DEVICE}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

    train_iter = iter(train_loader)
    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    print(f"\n🚀 Training for {TOTAL_STEPS} steps...\n")
    model.train()
    for step in range(TOTAL_STEPS):
        bx, by = get_batch()
        bx, by = bx.to(DEVICE), by.to(DEVICE)

        loss = loss_fn(model(bx).reshape(-1, VOCAB_SIZE), by.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
            vl = estimate_val_loss(model, val_loader, loss_fn)
            print(f"Step {step:4d} | Train: {loss.item():.4f} | Val: {vl:.4f} | LR: {lr:.6f}")

    torch.save(model.state_dict(), 'babygpt_final.pth')
    print(f"\n✅ Done! Final loss: {loss.item():.4f}")
    print("💾 Saved: babygpt_final.pth")


if __name__ == '__main__':
    main()