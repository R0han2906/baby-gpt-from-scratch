<div align="center"> <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2500&pause=800&color=FF6B35&center=true&vCenter=true&width=700&lines=BabyGPT+from+Scratch;LLaMA+3+Architecture+in+Pure+PyTorch;5.7M+Params+%7C+Shakespeare+Generation;RoPE+%7C+SwiGLU+%7C+GQA+%7C+KV+Cache" alt="BabyGPT" />


![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Params](https://img.shields.io/badge/Params-5.7M-FF6B35?style=for-the-badge&logo=brain&logoColor=white)
![Loss](https://img.shields.io/badge/Val_Loss-1.167-00C853?style=for-the-badge)


> *Built from scratch. No Karpathy. No HuggingFace. No copy-paste.*
*Every line of math, every tensor operation — handwritten.*

</div>

---

## ⚡ What is this?

A **decoder-only Transformer language model** built entirely from scratch in PyTorch, implementing the **same architecture as LLaMA 3 and Mistral** — trained on Tiny Shakespeare (1.1M characters) to generate coherent multi-character dialogue.

| Input | Output |
| --- | --- |
| `ROMEO:` | `My meaning woe, my lord, I would they fight,`<br>`Lest in thy brother, and unriversation`<br>`When may we return your own draw to yourself.` |

---

## 🏗️ Architecture

```
               ┌─────────────────────┐
               │   Input Tokens      │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Token Embedding    │
               │  + RMSNorm          │
               └──────────┬──────────┘
                          │
         ┌────────────────▼────────────────┐
         │                                 │
         │    Transformer Block × 6        │
         │    ┌───────────────────────┐    │
         │    │  RMSNorm              │    │
         │    │  ↓                    │    │
         │    │  GQA (8Q : 2KV)       │    │
         │    │  + RoPE               │    │
         │    │  + Sliding Window     │    │
         │    │  + KV Cache           │    │
         │    │  ↓                    │    │
         │    │  RMSNorm              │    │
         │    │  ↓                    │    │
         │    │  SwiGLU FFN           │    │
         │    └───────────────────────┘    │
         │                                 │
         └────────────────┬────────────────┘
                          │
               ┌──────────▼──────────┐
               │  Final RMSNorm      │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Linear Head        │
               │  (Weight Tied)      │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Logits (65 vocab)  │
               └─────────────────────┘
```

---

## 🧬 2024 Architecture Features

| Feature | Paper | What It Does |
| --- | --- | --- |
| **RoPE** | [Su et al. 2021](https://arxiv.org/abs/2104.09864) | Rotary position encoding — relative positions via rotation |
| **RMSNorm** | [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) | Normalization without mean subtraction — faster than LayerNorm |
| **SwiGLU** | [Shazeer 2020](https://arxiv.org/abs/2002.05202) | Gated FFN with Swish activation — used in LLaMA, PaLM |
| **GQA** | [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) | 8 Query heads share 2 KV heads — 4× memory savings |
| **SWA** | [Beltagy et al. 2020](https://arxiv.org/abs/2004.05150) | Attention limited to 128-token window — Mistral style |
| **Weight Tying** | [Press & Wolf 2017](https://arxiv.org/abs/1608.05859) | Embedding = Output weights — fewer params, better generalization |
| **KV Cache** | Standard | Cache past Keys/Values during inference — no recomputation |

---

## 📊 Training

| Hyperparameter | Value |
| --- | --- |
| Parameters | 5,739,520 |
| Dataset | Tiny Shakespeare (1.1M chars) |
| Vocab | 65 characters |
| Context Length | 128 tokens |
| Batch Size | 64 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning Rate | 5e-4 → 5e-5 (Cosine + 300 warmup) |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Dropout | 0.1 |
| Training Steps | 8,000 |
| Hardware | Single T4 GPU (Kaggle) |
| Training Time | ~10 minutes |

### Loss Curve

```
Loss
2.5 │██
    │ ██
2.0 │ ███
    │ ████
1.5 │ ██████
    │ ████████
1.0 │ █████████████████
    │
    └──────────────────────────────────────────── Steps
      0 1k 2k 3k 4k 5k 6k 7k 8k
```

**Final Train:** 1.155 | **Final Val:** 1.167

---

## 🎭 Sample Outputs

### KV Cache Generation (Fast)

> **ROMEO:**
In the sorrow could, who dost thou tell?
Where we he shall the term the matter of the state;
And then, thou know'st the gods sing like a world.**BRAKENBURY:**
Read to me; my noble Capulet commandment.**CAPULET:**
What can smile then?**KING:**
What says the queen of the people's death!
Sawing the stone, nor then fearful watch, right.
Say you, here comes the discovery.**Nurse:**
This that he shall, and well the world shall post:
God grant to the king, in the grave of heaven.**HAMLET:**
How fares the in his head of Buckingham,
As I do disperse to the tongue of blood?**PETRUCHIO:**
I, sir, it is so much an amorous reasons.
Then, by this woeful man, I will do attend it.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/baby-gpt-from-scratch.git
cd baby-gpt-from-scratch

# Install
pip install -r requirements.txt

# Train (~10 min on GPU)
python train.py

# Generate
python generate.py
```

---

## 🆚 Comparison with nanoGPT

|  | Karpathy nanoGPT | BabyGPT (Ours) |
| --- | --- | --- |
| Positional Encoding | Sinusoidal (2017) | RoPE (2021) |
| Normalization | LayerNorm | RMSNorm |
| Activation | GELU | SwiGLU |
| Attention | Multi-Head (MHA) | Grouped-Query (GQA) |
| Attention Window | Full | Sliding Window |
| Weight Tying | Optional | Yes |
| KV Cache | ❌ | ✅ with causal prefill |
| Parameters | ~10M | 5.7M |
| Output Quality | Good | Equal or better |

> **Half the parameters. Twice the architecture. Same quality. 💪**

---

## 🐛 Bugs Squashed (The Real Learning)

This project took 6 iterations to get right. Here's what broke and why:

| Version | Bug | Root Cause | Fix |
| --- | --- | --- | --- |
| V3 | GQA shape crash | RoPE(d_model) applied to K with kv_dim | Separate RoPE(kv_dim) for K |
| V4 | Double RoPE | RoPE in Embedding AND Attention | RoPE only in Attention |
| V4 | Cache overflow | Cache grew beyond training window | Trim to window_size |
| V5 | Position freeze | past_len = cache.size ≠ real position | Explicit seq_pos tracker |
| V5 | RoPE OOD | Model trained on pos 0-127, inference goes 200+ | Fallback to slow gen at 128 |
| V6 | Prefill leak | No causal mask during seed processing | Causal mask when T > 1 |

---

## 📁 Project Structure

```
baby-gpt-from-scratch/
├── babygpt.py          # Model architecture (all components)
├── train.py            # Training script
├── generate.py         # Generation (slow + KV cache)
├── requirements.txt    # Dependencies
├── README.md           # This file
└── LICENSE             # MIT
```

---

## 🔮 What's Next?

- BPE Tokenizer (replace char-level)

- Flash Attention (O(N) memory)

- Mixed Precision (fp16/bf16)

- torch.compile (20-30% speedup)

- Scale to 100M+ params

- RAG Pipeline integration

- LoRA Fine-tuning on custom data

- FastAPI deployment + Docker

---

## 📚 References

- Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)

- Zhang & Sennrich — Root Mean Square Layer Normalization (2019)

- Shazeer — GLU Variants Improve Transformer (2020)

- Ainslie et al. — GQA: Training Generalized Multi-Query Transformer (2023)

- Beltagy et al. — Longformer: The Long-Document Transformer (2020)

- Press & Wolf — Using the Output Embedding to Improve Language Models (2017)

- Loshchilov & Hutter — Decoupled Weight Decay Regularization (2019)

<div align="center">

Built with ❤️ and pure PyTorch tensors

If this helped you understand Transformers, drop a ⭐

<img src="https://komarev.com/ghpvc/?username=YOUR_USERNAME&repo=baby-gpt-from-scratch&color=FF6B35&style=for-the-badge" alt="views" /> </div>
