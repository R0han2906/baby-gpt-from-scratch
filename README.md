🧠 BabyGPT: Decoder-only Transformer from Scratch
<p align="center"> <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=2500&pause=1000&color=FF6B35&center=true&vCenter=true&width=700&lines=LLaMA+3+Architecture+in+Pure+PyTorch;5.7M+Params+%7C+Shakespeare+Generation;RoPE+%7C+SwiGLU+%7C+GQA+%7C+KV+Cache" alt="Typing SVG" /> </p><p align="center"> <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" /> <img src="https://img.shields.io/badge/Params-5.7M-FF6B35?style=for-the-badge&logo=cpu&logoColor=white" /> <img src="https://img.shields.io/badge/Val_Loss-1.167-00C853?style=for-the-badge&logo=checkmarx&logoColor=white" /> <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" /> </p>
Built from scratch. No Karpathy nanoGPT fork. No HuggingFace abstractions. Every tensor operation, math logic, and cache mechanism written by hand to deeply understand modern LLMs.

⚡ What is this?
A decoder-only Transformer language model built entirely from scratch in PyTorch, implementing the same architectural components as LLaMA 3 and Mistral. Trained on Tiny Shakespeare (1.1M characters) to generate coherent multi-character dialogue.

text

Input:  "ROMEO:"
Output: "My meaning woe, my lord, I would they fight,
         Lest in thy brother, and unriversation
         When may we return your own draw to yourself."
🏗️ Architecture Design (Interactive)
The model follows a modern Pre-Norm architecture with Grouped-Query Attention and SwiGLU.

mermaid

graph TD
    classDef core fill:#2b3137,stroke:#ff6b35,stroke-width:2px,color:#fff;
    classDef component fill:#1e1e1e,stroke:#4caf50,stroke-width:1px,color:#fff;
    
    A[Input Tokens]:::core --> B[Token Embedding + RMSNorm]:::component
    B --> C{Transformer Blocks x 6}:::core
    
    subgraph Transformer Block [6 Layers]
    C --> D[RMSNorm]:::component
    D --> E[GQA Attention + RoPE]:::component
    E -.->|KV Cache + Sliding Window| E
    E --> F[RMSNorm]:::component
    F --> G[SwiGLU FFN]:::component
    end
    
    G --> H[Final RMSNorm]:::component
    H --> I[Weight-Tied Linear Head]:::core
    I --> J[Logits Output]:::core
🧬 2024 Industry-Standard Features
Component	Paper / Inspiration	Why it's used?
RoPE	Su et al. 2021	Replaced absolute embeddings. Allows relative position tracking via rotation.
RMSNorm	Zhang & Sennrich 2019	No mean subtraction. Computationally faster than standard LayerNorm.
SwiGLU	Shazeer 2020	Two-branch gated FFN. Outperforms GELU/ReLU (Used in LLaMA, PaLM).
GQA	Ainslie et al. 2023	8 Query heads share 2 KV heads. Drastically reduces KV cache memory footprint.
SWA	Beltagy et al. 2020	Sliding Window Attention limits context to 128 tokens for efficiency (Mistral style).
Weight Tying	Press & Wolf 2017	Input embeddings and Output head share weights. Saves params, improves generalization.
KV Cache	Standard Practice	Autoregressive inference speedup with causal prefill masking.
🆚 Benchmark vs. nanoGPT
Feature	Karpathy nanoGPT	BabyGPT (Ours)
Positional Encoding	Sinusoidal (2017)	RoPE (2021)
Normalization	LayerNorm	RMSNorm
Activation	GELU	SwiGLU
Attention	Multi-Head (MHA)	Grouped-Query (GQA)
Attention Span	Full Sequence	Sliding Window
Inference	Recomputes everything	KV Cache integrated
Parameters	~10.0M	5.7M (40% smaller)
Output Quality	Good	Equal or Better 🏆
🎭 Generation Showcase (KV Cache)
The model learned character names, dialogue structure, and Elizabethan English style perfectly.

ROMEO:
In the sorrow could, who dost thou tell?
Where we he shall the term the matter of the state;
And then, thou know'st the gods sing like a world.

BRAKENBURY:
Read to me; my noble Capulet commandment.

KING:
What says the queen of the people's death!
Sawing the stone, nor then fearful watch, right.
Say you, here comes the discovery.

Nurse:
This that he shall, and well the world shall post:
God grant to the king, in the grave of heaven.

🚀 Quick Start
1. Clone & Install
Bash

git clone https://github.com/YOUR_USERNAME/baby-gpt-from-scratch.git
cd baby-gpt-from-scratch
pip install -r requirements.txt
2. Train the Model (~10 mins on T4 GPU)
Bash

python train.py
3. Generate Text (Compares Slow vs Fast KV Cache)
Bash

python generate.py
🐛 The Engineering Post-Mortem (Bugs Squashed)
Building this from scratch wasn't easy. Here are the major architectural bugs encountered and fixed during the 6 iterations:

GQA Shape Mismatch: RoPE(d_model) was accidentally applied to Keys (which have kv_dim).
Fix: Created separate RoPE(kv_dim) for K.
Double Rotation Anomaly: Applied RoPE in TokenEmbedding AND CausalSelfAttention. Training worked, but inference broke symmetrically.
Fix: Standardized RoPE strictly inside Attention.
Cache Overflow: Cache grew indefinitely beyond the training window_size (64), causing out-of-distribution positions.
Fix: Explicit cache trimming K[:, :, -self.window_size:].
RoPE Position Freeze: Used cache length as sequence position post-trimming, causing rotations to freeze at position 128.
Fix: Implemented explicit seq_pos state tracking.
Prefill Causal Leak: Seed prompt processing lacked a causal mask, allowing token 0 to attend to token 5.
Fix: Added dynamic causal masking when T > 1 during KV cache prefill.
📁 Repository Structure
text

baby-gpt-from-scratch/
├── babygpt.py          # Pure architecture (Classes: RMSNorm, RoPE, GQA, SwiGLU)
├── train.py            # Dataset loading, AdamW, Cosine LR, Training Loop
├── generate.py         # Autoregressive generation (w/ and w/o KV Cache)
├── requirements.txt    # Minimal dependencies (PyTorch)
└── README.md           # You are here
🔮 Roadmap / Next Steps
 Implement BPE Tokenizer (Tiktoken) to replace char-level tokenization.
 Integrate Flash Attention 2 for O(N) memory efficiency.
 Add Mixed Precision (fp16/bf16) for faster training.
 Scale up to 100M+ parameters.
 Wrap in FastAPI for real-time generation endpoint.
<p align="center"> <b>Built with ❤️ and pure PyTorch tensors</b><br><br> <i>If this repo helped you understand modern LLM architectures, drop a ⭐!</i> </p>
