import argparse
import math
import os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

PAD_IDX = 1  # matches dataset.py


import re as _re

def _detokenize(s: str) -> str:
    # undo spaCy tokenization spacing so sacrebleu doesn't complain
    s = _re.sub(r" ([.,!?;:'\")\]>])", r"\1", s)
    s = _re.sub(r"([({\[<]) ", r"\1", s)
    return s


def _corpus_bleu(hypotheses: list, references: list) -> float:
    # fallback chain: sacrebleu -> nltk -> 1-gram precision
    try:
        from sacrebleu.metrics import BLEU as SBL
        return SBL(tokenize='none').corpus_score(hypotheses, [references]).score
    except Exception:
        pass
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [[r.split()] for r in references]
        hyps_tok = [h.split()   for h in hypotheses]
        return corpus_bleu(refs_tok, hyps_tok, smoothing_function=SmoothingFunction().method3) * 100.0
    except Exception:
        pass
    from collections import Counter
    match = total = 0
    for h, r in zip(hypotheses, references):
        hc, rc = Counter(h.split()), Counter(r.split())
        match += sum(min(v, rc[k]) for k, v in hc.items())
        total += len(h.split())
    return 100.0 * match / max(total, 1)


class LabelSmoothingLoss(nn.Module):
    # y_smooth = (1-eps)*one_hot(y) + eps/(V-2)
    # last_confidence tracks mean softmax prob on the correct token (W&B Exp 5)

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size      = vocab_size
        self.pad_idx         = pad_idx
        self.smoothing       = smoothing
        self.last_confidence = 0.0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N, V      = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_val = self.smoothing / max(V - 2, 1)
            dist = logits.new_full((N, V), smooth_val)
            dist[:, self.pad_idx] = 0.0
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            dist[target == self.pad_idx] = 0.0

        loss    = -(dist * log_probs).sum()
        non_pad = (target != self.pad_idx).sum().clamp(min=1)

        # track prediction confidence for Exp 5
        with torch.no_grad():
            mask = target != self.pad_idx
            if mask.any():
                probs = F.softmax(logits, dim=-1)
                self.last_confidence = probs[mask, target[mask]].mean().item()

        return loss / non_pad


def run_epoch(
    data_iter,
    model:     Transformer,
    loss_fn:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler  = None,
    epoch_num: int  = 0,
    is_train:  bool = True,
    device:    str  = "cpu",
    *,
    global_step:      int  = 0,
    track_grad_norms: bool = False,  # W&B Exp 2: log ||grad W_q||, ||grad W_k||
    log_confidence:   bool = False,  # W&B Exp 5: per-step prediction confidence
    wandb_prefix:     str  = "",
) -> float:
    # one epoch; returns mean batch loss
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    desc = f"{'Train' if is_train else 'Val':5s} epoch {epoch_num}"

    with torch.set_grad_enabled(is_train):
        for batch_idx, (src, tgt) in enumerate(tqdm(data_iter, desc=desc, leave=False)):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src, PAD_IDX)
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            B, T, V = logits.shape
            loss = loss_fn(logits.contiguous().view(-1, V), tgt_out.contiguous().view(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # gradient norms for first 1000 steps (Exp 2)
                if track_grad_norms and _WANDB_OK:
                    step = global_step + batch_idx
                    if step < 1000:
                        gn = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None and ('W_q' in name or 'W_k' in name):
                                tag = name.replace('.', '/').replace('weight', 'w')
                                gn[f"grad_norm/{tag}"] = param.grad.norm().item()
                        if gn:
                            wandb.log(gn, step=step)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if _WANDB_OK:
                    step = global_step + batch_idx
                    log  = {
                        f"{wandb_prefix}loss": loss.item(),
                        f"{wandb_prefix}lr":   optimizer.param_groups[0]['lr'],
                    }
                    if log_confidence and hasattr(loss_fn, 'last_confidence'):
                        log[f"{wandb_prefix}pred_confidence"] = loss_fn.last_confidence
                    wandb.log(log, step=step)

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def greedy_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    # token-by-token greedy decode; returns [1, out_len]
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break
    return ys


def greedy_decode_batch(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    # decode an entire batch in parallel; ~10-50x faster than per-sentence loop
    model.eval()
    B = src.size(0)
    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # pre-allocate to avoid cat-growing tensors every step (prevents OOM fragmentation)
        ys   = torch.full((B, max_len), end_symbol, dtype=torch.long, device=device)
        ys[:, 0] = start_symbol

        eos_fill = torch.full((B,), end_symbol, dtype=torch.long, device=device)
        done     = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            cur_ys   = ys[:, :t]
            tgt_mask = make_tgt_mask(cur_ys, PAD_IDX)
            logits   = model.decode(memory, src_mask, cur_ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1)

            next_tok = torch.where(done, eos_fill, next_tok)
            ys[:, t] = next_tok
            done     = done | (next_tok == end_symbol)

            if done.all():
                ys = ys[:, :t + 1]
                break

    return ys


def evaluate_bleu(
    model:           Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device:          str = "cpu",
    max_len:         int = 100,
) -> float:
    # corpus-level BLEU (0-100); uses batched greedy decode for speed
    SOS, EOS = 2, 3
    SKIP = {'<sos>', '<eos>', '<pad>'}

    model.eval()
    hypotheses, references = [], []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src      = src.to(device)
            src_mask = make_src_mask(src, PAD_IDX)

            preds = greedy_decode_batch(model, src, src_mask, max_len, SOS, EOS, device)

            for i in range(src.size(0)):
                hyp = []
                for idx in preds[i, 1:]:
                    tok = tgt_vocab.lookup_token(idx.item())
                    if tok == '<eos>': break
                    if tok not in SKIP: hyp.append(tok)
                hypotheses.append(' '.join(hyp))

                ref = []
                for idx in tgt[i, 1:]:
                    tok = tgt_vocab.lookup_token(idx.item())
                    if tok == '<eos>': break
                    if tok not in SKIP: ref.append(tok)
                references.append(' '.join(ref))

    hypotheses = [_detokenize(h) for h in hypotheses]
    references  = [_detokenize(r) for r in references]
    return _corpus_bleu(hypotheses, references)


def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': {
                'src_vocab_size': model.src_embedding.num_embeddings,
                'tgt_vocab_size': model.tgt_embedding.num_embeddings,
                'd_model':        model.d_model,
                'N':              len(model.encoder.layers),
                'num_heads':      model.encoder.layers[0].self_attn.num_heads,
                'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
                'dropout':        model.encoder.layers[0].dropout.p,
                'pad_idx':        model.pad_idx,
                'pe_type':        model.pe_type,
                'use_scale':      model.use_scale,
            },
            'src_vocab': model.src_vocab,
            'tgt_vocab': model.tgt_vocab,
        },
        path,
    )


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler                                  = None,
) -> int:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'src_vocab' in ckpt:
        model.src_vocab = ckpt['src_vocab']
    if 'tgt_vocab' in ckpt:
        model.tgt_vocab = ckpt['tgt_vocab']
    return int(ckpt.get('epoch', 0))


def log_attention_heatmaps(model, dataset, device="cpu", sample_idx=0, wandb_step=0) -> None:
    # log per-head attention heatmaps from the last encoder layer (W&B Exp 3)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    src_ids, _ = dataset[sample_idx]
    src_ids    = src_ids.unsqueeze(0).to(device)
    src_mask   = make_src_mask(src_ids, PAD_IDX)

    with torch.no_grad():
        model.encode(src_ids, src_mask)

    attn      = model.encoder.layers[-1].self_attn.attn_weights[0]  # [h, L, L]
    num_heads = attn.size(0)
    src_toks  = ['<sos>'] + dataset.src_toks[sample_idx] + ['<eos>']
    L         = min(attn.size(1), len(src_toks))

    cols = 4
    rows = math.ceil(num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for h in range(num_heads):
        ax   = axes_flat[h]
        data = attn[h, :L, :L].cpu().numpy()
        im   = ax.imshow(data, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
        ax.set_xticks(range(L))
        ax.set_xticklabels(src_toks[:L], rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(L))
        ax.set_yticklabels(src_toks[:L], fontsize=6)
        ax.set_title(f'Head {h}', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(num_heads, len(axes_flat)):
        axes_flat[h].axis('off')

    fig.suptitle('Last encoder layer — self-attention per head', fontsize=10)
    fig.tight_layout()

    if _WANDB_OK:
        wandb.log({'attention/encoder_last_layer': wandb.Image(fig)}, step=wandb_step)
    plt.close(fig)


# default config used as the baseline
_BASE_CFG = dict(
    d_model       = 256,
    N             = 3,
    num_heads     = 8,
    d_ff          = 1024,
    dropout       = 0.3,
    batch_size    = 128,
    num_epochs    = 50,
    warmup_steps  = 4000,
    scheduler_type   = 'noam',
    fixed_lr         = 1e-4,
    pe_type          = 'sinusoidal',
    use_scale        = True,
    smoothing        = 0.1,
    track_grad_norms = False,
    log_confidence   = False,
    log_attention    = False,
)


def _run_one_experiment(cfg: dict) -> float:
    from dataset import Multi30kDataset, collate_fn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}\nExperiment : {cfg['name']}\nDevice     : {device}\n{'='*60}")

    train_ds = Multi30kDataset('train')
    val_ds   = Multi30kDataset('val',  src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset('test', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    _coll = partial(collate_fn, pad_idx=PAD_IDX)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  collate_fn=_coll, num_workers=0, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, collate_fn=_coll, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, collate_fn=_coll, num_workers=0)

    # smaller batch for BLEU eval to avoid OOM during decoding
    bleu_batch = min(32, cfg['batch_size'])
    bleu_val_loader  = DataLoader(val_ds,  batch_size=bleu_batch, shuffle=False, collate_fn=_coll, num_workers=0)
    bleu_test_loader = DataLoader(test_ds, batch_size=bleu_batch, shuffle=False, collate_fn=_coll, num_workers=0)

    model = Transformer(
        src_vocab_size = len(train_ds.src_vocab),
        tgt_vocab_size = len(train_ds.tgt_vocab),
        d_model        = cfg['d_model'],
        N              = cfg['N'],
        num_heads      = cfg['num_heads'],
        d_ff           = cfg['d_ff'],
        dropout        = cfg['dropout'],
        pe_type        = cfg['pe_type'],
        use_scale      = cfg['use_scale'],
        src_vocab      = train_ds.src_vocab,
        tgt_vocab      = train_ds.tgt_vocab,
    ).to(device)
    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Adam with paper betas; base lr=1.0 for Noam, actual lr for fixed
    base_lr   = 1.0 if cfg['scheduler_type'] == 'noam' else cfg.get('fixed_lr', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)

    if cfg['scheduler_type'] == 'noam':
        scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(train_ds.tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = cfg['smoothing'],
    )

    if _WANDB_OK:
        try:
            wandb.init(project="da6401-a3", name=cfg['name'], config=cfg, reinit="finish_previous")
            wandb.watch(model, log='gradients', log_freq=200)
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    best_bleu   = -1.0
    global_step = 0
    ckpt_path   = cfg.get('ckpt_path', 'checkpoint.pt')

    for epoch in range(cfg['num_epochs']):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num        = epoch,
            is_train         = True,
            device           = device,
            global_step      = global_step,
            track_grad_norms = cfg.get('track_grad_norms', False),
            log_confidence   = cfg.get('log_confidence', False),
            wandb_prefix     = "train/",
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num    = epoch,
            is_train     = False,
            device       = device,
            wandb_prefix = "val/",
        )
        if device == 'cuda':
            torch.cuda.empty_cache()
        val_bleu = evaluate_bleu(model, bleu_val_loader, train_ds.tgt_vocab, device)

        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  BLEU={val_bleu:.2f}")

        global_step += len(train_loader)

        if _WANDB_OK:
            try:
                wandb.log({
                    'epoch':            epoch,
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss':   val_loss,
                    'val/bleu':         val_bleu,
                }, step=global_step - 1)
            except Exception:
                pass

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)
            if ckpt_path != 'checkpoint.pt':
                save_checkpoint(model, optimizer, scheduler, epoch, 'checkpoint.pt')
            print(f"  * saved  (val BLEU {best_bleu:.2f})")

    # attention heatmaps if requested (Exp 3)
    if cfg.get('log_attention', False):
        load_checkpoint(ckpt_path, model)
        log_attention_heatmaps(model, train_ds, device, sample_idx=0, wandb_step=global_step)

    load_checkpoint(ckpt_path, model)
    if device == 'cuda':
        torch.cuda.empty_cache()
    test_bleu = evaluate_bleu(model, bleu_test_loader, train_ds.tgt_vocab, device)
    print(f"\n  Test BLEU = {test_bleu:.2f}")

    if _WANDB_OK:
        try:
            wandb.log({'test/bleu': test_bleu}, step=global_step)
            wandb.finish()
        except Exception:
            pass

    return test_bleu


def run_training_experiment() -> None:
    parser = argparse.ArgumentParser(description="DA6401-A3 experiment runner")
    parser.add_argument(
        '--exp', default='baseline',
        choices=['baseline', 'noam_vs_fixed', 'scaling_ablation',
                 'attention_rollout', 'pe_ablation', 'label_smoothing', 'all'],
    )
    args, _ = parser.parse_known_args()

    def run(**overrides):
        cfg = dict(_BASE_CFG)
        cfg.update(overrides)
        return _run_one_experiment(cfg)

    exp = args.exp

    if exp in ('baseline', 'attention_rollout', 'all'):
        run(name='baseline', log_attention=exp in ('attention_rollout', 'all'), ckpt_path='checkpoint_baseline.pt')

    if exp in ('noam_vs_fixed', 'all'):
        run(name='fixed_lr_1e4', scheduler_type='fixed', fixed_lr=1e-4, ckpt_path='checkpoint_fixed_lr.pt')
        if exp == 'noam_vs_fixed':
            run(name='noam_cmp', ckpt_path='checkpoint_noam_cmp.pt')

    if exp in ('scaling_ablation', 'all'):
        run(name='with_scale', track_grad_norms=True, ckpt_path='checkpoint_with_scale.pt')
        run(name='no_scale',   use_scale=False, track_grad_norms=True, ckpt_path='checkpoint_no_scale.pt')

    if exp in ('pe_ablation', 'all'):
        run(name='sinusoidal_pe', pe_type='sinusoidal', ckpt_path='checkpoint_sin_pe.pt')
        run(name='learned_pe',    pe_type='learned',    ckpt_path='checkpoint_learned_pe.pt')

    if exp in ('label_smoothing', 'all'):
        run(name='smooth_0.1', smoothing=0.1, log_confidence=True, ckpt_path='checkpoint_smooth01.pt')
        run(name='smooth_0.0', smoothing=0.0, log_confidence=True, ckpt_path='checkpoint_smooth00.pt')

    print("\nAll experiments finished.")


if __name__ == "__main__":
    run_training_experiment()
