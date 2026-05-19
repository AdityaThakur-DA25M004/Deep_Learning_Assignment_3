# DA6401 — Assignment 3: Transformer for Machine Translation

> **"Attention Is All You Need"** — Vaswani et al., 2017  
> German → English Neural Machine Translation on the Multi30k dataset

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📊 W&B Report | [View Report](https://wandb.ai/da25m004-iit-madras/da6401-a3/reports/-Assignment-3-Report--VmlldzoxNjkyNTI2OA?accessToken=w05wtz1m7kcb2k3gao2l5e9715v31urh8dr5yn68g8dsp51dv4j13h7j191nja9a) |
| 💻 GitHub Repository | [View Repository](https://github.com/AdityaThakur-DA25M004/Deep_Learning_Assignment_3.git) |
| 📄 Base Paper | [Attention Is All You Need (NeurIPS 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) |
| 🤗 Dataset | [bentrevett/multi30k on HuggingFace](https://huggingface.co/datasets/bentrevett/multi30k) |
| 🏫 Assignment Skeleton | [MiRL-IITM/da6401_assignment_3](https://github.com/MiRL-IITM/da6401_assignment_3) |

---

## 📁 Project Structure

```
assignment3/
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
├── dataset.py             ← Multi30k loading, tokenisation, Vocab
├── lr_scheduler.py        ← Noam warm-up LR scheduler
├── model.py               ← Core Transformer architecture
├── train.py               ← Training loop, greedy decoding, BLEU eval
└── W_b_train.py           ← Extended trainer with W&B ablation experiments
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/da6401_assignment_3.git
cd da6401_assignment_3
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy language models

```bash
python -m spacy download de_core_news_sm   # German tokeniser
python -m spacy download en_core_web_sm    # English tokeniser
```

### 4. Login to Weights & Biases

```bash
wandb login
```

---

## 🗃️ Dataset

The **[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k)** dataset is automatically downloaded from HuggingFace on first run and cached locally under `data/`:

| Split | Pairs |
|-------|-------|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

Manual download can be triggered by instantiating the dataset:

```python
from dataset import Multi30kDataset
train_ds = Multi30kDataset(split="train")   # downloads & caches automatically
```

---

## 🏗️ Implementation Details

### Task 1 — Scaled Dot-Product & Multi-Head Attention (`model.py`)

Implements the attention mechanism from [§3.2 of the paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf):

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
MultiHead(Q,K,V)   = Concat(head₁, …, headₕ) · Wᴼ
```

- Masking: padding mask `[B, 1, 1, T]` for the encoder; combined padding + causal mask `[B, 1, T, T]` for the decoder.
- `torch.nn.MultiheadAttention` is **not** used — attention is implemented from scratch.

### Task 2 — Encoder & Decoder Stacks (`model.py`)

| Component | Details |
|-----------|---------|
| Positional Encoding | Sinusoidal (sin/cos), pre-computed buffer, **not** a trainable parameter |
| Add & Norm | Post-LayerNorm — matching the original paper spec |
| Feed-Forward | `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` |
| Encoder | N identical `EncoderLayer` modules + final LayerNorm |
| Decoder | N identical `DecoderLayer` modules + final LayerNorm |

### Task 3 — Training Pipeline (`train.py` / `W_b_train.py`)

| Component | Details |
|-----------|---------|
| Loss | Label Smoothing (ε = 0.1), pad positions receive 0 probability |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| LR Schedule | Noam warm-up (`warmup_steps = 4000`) |
| Grad Clipping | `max_norm = 1.0` |
| Inference | Autoregressive greedy decoding |
| Evaluation | Corpus-level BLEU via [sacrebleu](https://github.com/mjpost/sacrebleu) → nltk → 1-gram precision fallback |

---

## 📐 Noam LR Scheduler (`lr_scheduler.py`)

Implements the schedule from [§5.3 of the paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf):

```
lrate = d_model^(-0.5) · min(step^(-0.5),  step · warmup_steps^(-1.5))
```

Visualise the schedule:

```bash
python lr_scheduler.py
```

---

## 🚀 Training

### Autograder / Submission Model

Run with the tuned hyperparameters used for the submitted checkpoint:

```bash
python train.py
```

#### Autograder Hyperparameters

These are the exact values used for the best checkpoint submitted for autograder evaluation (scaled down from the paper defaults to suit the smaller Multi30k dataset):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | **256** | Smaller than paper's 512 — fits Multi30k well |
| `N` | **3** | 3 encoder + 3 decoder layers (paper uses 6) |
| `num_heads` | **8** | 8 attention heads → d_k = 32 |
| `d_ff` | **512** | FFN inner dimension (paper uses 2048) |
| `dropout` | **0.35** | Higher dropout to regularise the smaller model |
| `label_smoothing` | **0.1** | As per paper |
| `warmup_steps` | **4000** | As per paper |
| `batch_size` | **128** | |
| `num_epochs` | **70** | With early stopping (patience = 8) |
| `grad_clip` | **1.0** | `nn.utils.clip_grad_norm_` |
| `use_noam` | **True** | Noam schedule active |
| `fixed_lr` | `1e-4` | Fallback only if `use_noam=False` |
| `use_bucketed_sampler` | **True** | Groups similar-length sequences to reduce padding waste |
| `word_dropout` | **0.0** | Off in the submitted config |
| `use_ema` | **False** | Off in the submitted config |
| `eval_use_beam` | **False** | Greedy decoding for evaluation |
| `eval_max_len` | **150** | Max decode length at inference |
| `use_learned_pos_embed` | **False** | Sinusoidal PE used |
| `seed` | **42** | |
| `early_stop_patience` | **8** | Stops if val BLEU doesn't improve for 8 epochs |

### W&B Experiments

Run all five ablation experiments:

```bash
python W_b_train.py --exp all
```

Or run a specific experiment:

```bash
python W_b_train.py --exp baseline           # baseline model + attention heatmaps
python W_b_train.py --exp noam_vs_fixed      # Exp 1: Noam vs fixed LR
python W_b_train.py --exp scaling_ablation   # Exp 2: with/without 1/√dₖ scaling
python W_b_train.py --exp attention_rollout  # Exp 3: per-head attention heatmaps
python W_b_train.py --exp pe_ablation        # Exp 4: sinusoidal vs learned PE
python W_b_train.py --exp label_smoothing    # Exp 5: ε=0.1 vs ε=0.0
```

#### W&B Base Config (`_BASE_CFG`)

All five experiments share this base config; each experiment applies its own overrides on top:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | **256** | |
| `N` | **3** | |
| `num_heads` | **8** | |
| `d_ff` | **1024** | Wider FFN than the autograder config |
| `dropout` | **0.3** | |
| `batch_size` | **128** | |
| `num_epochs` | **50** | |
| `warmup_steps` | **4000** | |
| `scheduler_type` | `'noam'` | Overridden to `'fixed'` in Exp 1 |
| `fixed_lr` | `1e-4` | Used when `scheduler_type='fixed'` |
| `smoothing` | **0.1** | Overridden to `0.0` in Exp 5 |
| `pe_type` | `'sinusoidal'` | Overridden to `'learned'` in Exp 4 |
| `use_scale` | **True** | Overridden to `False` in Exp 2 |
| `track_grad_norms` | `False` | Enabled in Exp 2 |
| `log_confidence` | `False` | Enabled in Exp 5 |
| `log_attention` | `False` | Enabled in Exp 3 |

---

## 📊 W&B Experiments (Report)

### Experiment 1 — Noam Scheduler vs. Fixed LR

Compares `noam_cmp` (Noam warm-up) against `fixed_lr_1e4` (constant LR = 10⁻⁴). Overlays training loss and validation BLEU curves and explains why the warm-up phase prevents early divergence of the self-attention weight matrices.

### Experiment 2 — Scaling Factor Ablation (1/√dₖ)

Trains `with_scale` and `no_scale` variants. Logs gradient norms of all `W_q` and `W_k` weight matrices during the first 1,000 steps (`track_grad_norms=True`) and relates findings to the vanishing-gradient argument in [§3.2.1 of the paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

### Experiment 3 — Attention Rollout & Head Specialisation

Extracts per-head attention weights from the last encoder layer (`log_attention=True`) and logs a full heatmap grid to W&B. Identifies heads performing local syntactic attention, positional shifting, and long-range dependency capture, and discusses head redundancy.

### Experiment 4 — Sinusoidal PE vs. Learned Embeddings

Compares `sinusoidal_pe` against `learned_pe` (`pe_type='learned'`, backed by `torch.nn.Embedding`). Reports validation BLEU for both and discusses sinusoidal PE's theoretical ability to generalise to sequence lengths unseen during training.

### Experiment 5 — Label Smoothing (ε = 0.1 vs. ε = 0.0)

Runs `smooth_0.1` and `smooth_0.0` with `log_confidence=True`, which logs the mean softmax probability assigned to the correct token at every training step. Explains label smoothing as a regulariser that prevents overconfidence while slightly increasing training perplexity.

---

## 💾 Checkpoint API

```python
from W_b_train import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, optimizer, scheduler, epoch=10, path="checkpoint.pt")

# Restore
epoch = load_checkpoint("checkpoint.pt", model, optimizer, scheduler)
```

Saved keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `model_config`, `src_vocab`, `tgt_vocab`.

---

## 🔍 Inference

```python
from w_b_model import Transformer
from W_b_train import load_checkpoint

model = Transformer(src_vocab_size=..., tgt_vocab_size=...)
load_checkpoint("checkpoint.pt", model)

translation = model.infer("Zwei Männer stehen auf einer Straße.")
print(translation)  # → "Two men are standing on a street."
```

---

## 📋 Autograder Contract

The following signatures must remain unmodified:

```python
# model.py
scaled_dot_product_attention(Q, K, V, mask)         → (output, attn_weights)
MultiHeadAttention.forward(q, k, v, mask)            → Tensor
PositionalEncoding.forward(x)                        → Tensor
make_src_mask(src, pad_idx)                          → BoolTensor  [B, 1, 1, T]
make_tgt_mask(tgt, pad_idx)                          → BoolTensor  [B, 1, T, T]
Transformer.encode(src, src_mask)                    → Tensor
Transformer.decode(memory, src_m, tgt, tgt_m)        → Tensor

# train.py
greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
evaluate_bleu(model, test_dataloader, tgt_vocab, device)  → float
save_checkpoint(model, optimizer, scheduler, epoch, path) → None
load_checkpoint(path, model, optimizer, scheduler)        → int
```

---

## 📦 Dependencies

```
torch
numpy
matplotlib
scikit-learn
wandb
datasets
spacy
tqdm
sacrebleu     # preferred BLEU backend; falls back to nltk then 1-gram precision
nltk          # optional fallback
```

Install all with:

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm en_core_web_sm
```

---

## 📝 Submission Checklist

- [ ] All `raise NotImplementedError` stubs implemented
- [ ] W&B report is **public** and the link is updated in this README
- [ ] GitHub repository is up to date
- [ ] Checkpoint accessible via Google Drive link embedded in `Transformer.__init__`
- [ ] Training and test splits strictly separated — no data leakage
- [ ] Submitted on [Gradescope](https://www.gradescope.com) before **19th May 2026, 23:59**

---

## 📖 References

1. Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). NeurIPS 2017.
2. Elliott, D. et al. (2016). [*Multi30K: Multilingual English-German Image Descriptions*](https://huggingface.co/datasets/bentrevett/multi30k).
3. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP.
4. [sacrebleu](https://github.com/mjpost/sacrebleu) — Reproducible BLEU scores.