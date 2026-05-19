from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    # spread eps mass evenly over non-pad vocab; keeps pad rows at zero

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_val = self.smoothing / max(self.vocab_size - 2, 1)
            true_dist  = torch.full_like(log_probs, smooth_val)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            true_dist[target == self.pad_idx] = 0.0

        per_token_loss = -(true_dist * log_probs).sum(dim=-1)
        n_tokens = (target != self.pad_idx).sum().clamp(min=1)
        return per_token_loss.sum() / n_tokens


class EMA:
    # exponential moving average of weights; opt-in, off by default

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        msd    = model.state_dict()
        backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


class LengthBucketedSampler(Sampler[List[int]]):
    # group similar-length sequences to cut down on padding waste

    def __init__(self, lengths, batch_size, shuffle=True, sort_size_mult=50, seed=42):
        self.lengths    = list(lengths)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.sort_size  = batch_size * sort_size_mult
        self.seed       = seed
        self.epoch      = 0
        self._batches: List[List[int]] = []
        self._build()

    def _build(self) -> None:
        idxs = list(range(len(self.lengths)))
        rng  = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idxs)
        batches: List[List[int]] = []
        for i in range(0, len(idxs), self.sort_size):
            chunk = idxs[i:i + self.sort_size]
            chunk.sort(key=lambda j: self.lengths[j])
            for k in range(0, len(chunk), self.batch_size):
                batches.append(chunk[k:k + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        self._batches = batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = 1,
    log_every: int = 100,
    grad_clip: float = 1.0,
    wandb_run=None,
    use_tqdm: bool = True,
    ema: Optional[EMA] = None,
) -> float:
    # one epoch; returns mean per-non-pad-token loss
    model.train(is_train)
    total_loss   = 0.0
    total_tokens = 0

    iterator = data_iter
    if use_tqdm:
        try:
            from tqdm import tqdm
            phase    = "train" if is_train else "val  "
            iterator = tqdm(data_iter, desc=f"{phase} epoch {epoch_num}", leave=False, ncols=100)
        except ImportError:
            pass

    for step, (src, tgt) in enumerate(iterator):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

        n_tok         = (tgt_output != pad_idx).sum().item()
        total_loss   += loss.item() * n_tok
        total_tokens += n_tok

        if is_train and wandb_run is not None and step % log_every == 0:
            wandb_run.log({
                "train/step_loss": loss.item(),
                "train/lr":        optimizer.param_groups[0]["lr"],
                "epoch":           epoch_num,
            })

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    # token-by-token greedy decode; returns [1, out_len]
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_len = ys.size(1)
            causal  = torch.triu(
                torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            logits     = model.decode(memory, src_mask, ys, causal)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
    return ys


def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    device: str = "cpu",
) -> torch.Tensor:
    # GNMT-style beam search; returns [1, out_len]
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory     = model.encode(src, src_mask)
        memory     = memory.expand(beam_size, -1, -1).contiguous()
        src_mask_b = src_mask.expand(beam_size, -1, -1, -1).contiguous()

        ys     = torch.full((beam_size, 1), start_symbol, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = float("-inf")

        finished: List[Tuple[float, torch.Tensor]] = []

        for _ in range(max_len - 1):
            tgt_len = ys.size(1)
            causal  = torch.triu(
                torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            logits    = model.decode(memory, src_mask_b, ys, causal)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            V    = log_probs.size(-1)
            cand = scores.unsqueeze(1) + log_probs
            flat = cand.reshape(-1)
            topk_scores, topk_idx = flat.topk(beam_size)

            beam_idx = torch.div(topk_idx, V, rounding_mode="floor")
            tok_idx  = topk_idx % V
            ys       = torch.cat([ys[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores   = topk_scores

            for b in range(beam_size):
                if tok_idx[b].item() == end_symbol:
                    seq_len = ys.size(1)
                    lp      = ((5.0 + seq_len) / 6.0) ** length_penalty
                    finished.append((scores[b].item() / lp, ys[b].clone()))
                    scores[b] = float("-inf")

            if torch.isinf(scores).all():
                break

        for b in range(beam_size):
            if not torch.isinf(scores[b]):
                seq_len = ys.size(1)
                lp      = ((5.0 + seq_len) / 6.0) ** length_penalty
                finished.append((scores[b].item() / lp, ys[b].clone()))

        if not finished:
            return ys[:1]

        best = max(finished, key=lambda x: x[0])[1]
        return best.unsqueeze(0)


def _itos_from_vocab(vocab) -> dict:
    if hasattr(vocab, "get_itos"):
        return {i: t for i, t in enumerate(vocab.get_itos())}
    if hasattr(vocab, "itos"):
        return {i: t for i, t in enumerate(vocab.itos)}
    if isinstance(vocab, dict):
        return {idx: tok for tok, idx in vocab.items()}
    raise TypeError(f"Unsupported vocab type: {type(vocab)}")


def _vocab_lookup(vocab, key, default=None):
    if isinstance(vocab, dict):
        return vocab.get(key, default)
    if hasattr(vocab, "get") and callable(getattr(vocab, "get")):
        try:
            return vocab.get(key, default)
        except TypeError:
            pass
    try:
        return vocab[key]
    except Exception:
        return default


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 150,
    src_pad_idx: int = 1,
    use_beam: bool = False,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> float:
    # corpus-level BLEU (0-100); greedy by default
    model.eval()
    itos    = _itos_from_vocab(tgt_vocab)
    sos_idx = _vocab_lookup(tgt_vocab, "<sos>", 2)
    eos_idx = _vocab_lookup(tgt_vocab, "<eos>", 3)
    pad_idx = _vocab_lookup(tgt_vocab, "<pad>", 1)

    def ids_to_tokens(ids):
        toks = []
        for i in ids:
            i = int(i)
            if i in (sos_idx, pad_idx):
                continue
            if i == eos_idx:
                break
            toks.append(itos.get(i, "<unk>"))
        return toks

    references, hypotheses = [], []
    try:
        from tqdm import tqdm
        loop = tqdm(test_dataloader, desc="BLEU eval", ncols=100, leave=False)
    except ImportError:
        loop = test_dataloader

    with torch.no_grad():
        for src, tgt in loop:
            src = src.to(device)
            tgt = tgt.to(device)
            for i in range(src.size(0)):
                src_i    = src[i:i + 1]
                src_mask = make_src_mask(src_i, pad_idx=src_pad_idx)

                if use_beam:
                    pred = beam_search_decode(
                        model, src_i, src_mask, max_len=max_len,
                        start_symbol=sos_idx, end_symbol=eos_idx,
                        beam_size=beam_size, length_penalty=length_penalty,
                        device=device,
                    )
                else:
                    pred = greedy_decode(
                        model, src_i, src_mask, max_len=max_len,
                        start_symbol=sos_idx, end_symbol=eos_idx,
                        device=device,
                    )

                hypotheses.append(ids_to_tokens(pred[0].cpu().tolist()))
                references.append(ids_to_tokens(tgt[i].cpu().tolist()))

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        score = corpus_bleu(
            [[r] for r in references], hypotheses,
            smoothing_function=SmoothingFunction().method1,
        )
        return float(score) * 100.0
    except ImportError:
        pass

    return _manual_corpus_bleu(hypotheses, references) * 100.0


def _manual_corpus_bleu(hypotheses, references, max_n: int = 4) -> float:
    # fallback BLEU-4 with brevity penalty (0..1)
    from collections import Counter
    weights = [1.0 / max_n] * max_n

    def ngrams(seq, n):
        return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    clipped = [0] * max_n
    totals  = [0] * max_n
    hyp_len_total = ref_len_total = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len_total += len(hyp)
        ref_len_total += len(ref)
        for n in range(1, max_n + 1):
            hyp_ng  = ngrams(hyp, n)
            ref_ng  = ngrams(ref, n)
            overlap = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            clipped[n - 1] += overlap
            totals[n - 1]  += sum(hyp_ng.values())

    precisions = [(clipped[i] / totals[i]) if totals[i] > 0 else 0.0 for i in range(max_n)]
    if min(precisions) == 0:
        return 0.0

    log_p = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bp    = 1.0 if hyp_len_total > ref_len_total else math.exp(
        1 - ref_len_total / max(hyp_len_total, 1)
    )
    return bp * math.exp(log_p)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab=None,
    tgt_vocab=None,
) -> None:
    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config":         model.get_config(),
    }
    if src_vocab is not None:
        payload["src_vocab"] = src_vocab
    if tgt_vocab is not None:
        payload["tgt_vocab"] = tgt_vocab
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if getattr(model, "tie_weights", False):
        try:
            model._tie_embeddings_and_generator()
        except Exception:
            pass
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[load_checkpoint] could not restore optimizer: {e}")
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[load_checkpoint] could not restore scheduler: {e}")
    if "src_vocab" in ckpt:
        model.src_vocab = ckpt["src_vocab"]
    if "tgt_vocab" in ckpt:
        model.tgt_vocab = ckpt["tgt_vocab"]
    return int(ckpt.get("epoch", 0))


def run_training_experiment(
    *,
    d_model: int = 256,
    N: int = 3,
    num_heads: int = 8,
    d_ff: int = 512,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    warmup_steps: int = 4000,
    batch_size: int = 128,
    num_epochs: int = 25,
    use_noam: bool = True,
    fixed_lr: float = 1e-4,
    grad_clip: float = 1.0,
    use_bucketed_sampler: bool = True,
    word_dropout: float = 0.0,
    use_ema: bool = False,
    ema_decay: float = 0.9995,
    eval_use_beam: bool = False,
    eval_beam_size: int = 4,
    eval_length_penalty: float = 0.6,
    project: str = "da6401-a3",
    run_name: Optional[str] = None,
    ckpt_dir: str = "checkpoints",
    seed: int = 42,
    early_stop_patience: int = 8,
    use_learned_pos_embed: bool = False,
    eval_max_len: int = 150,
) -> None:
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import Multi30kDataset, collate_fn, PAD_IDX
    from lr_scheduler import NoamScheduler

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=project, name=run_name,
            config=dict(
                d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
                dropout=dropout, label_smoothing=label_smoothing,
                warmup_steps=warmup_steps, batch_size=batch_size,
                num_epochs=num_epochs, use_noam=use_noam, fixed_lr=fixed_lr,
                early_stop_patience=early_stop_patience,
                word_dropout=word_dropout, use_ema=use_ema,
                eval_use_beam=eval_use_beam,
            ),
        )
    except ImportError:
        print("[train] wandb not installed - skipping logging.")

    print("[train] Loading datasets...")
    train_ds = Multi30kDataset(split="train", word_dropout=word_dropout)
    val_ds   = Multi30kDataset(split="val",  src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset(split="test", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    src_vsize = len(train_ds.src_vocab)
    tgt_vsize = len(train_ds.tgt_vocab)
    print(f"[train] |src vocab|={src_vsize}  |tgt vocab|={tgt_vsize}  "
          f"|train|={len(train_ds)}  |val|={len(val_ds)}  |test|={len(test_ds)}")

    if use_bucketed_sampler:
        train_lengths = [train_ds.src_len(i) for i in range(len(train_ds))]
        train_sampler = LengthBucketedSampler(train_lengths, batch_size=batch_size, shuffle=True, seed=seed)
        train_loader  = DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=0,
        )
    else:
        train_sampler = None
        train_loader  = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=0,
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    model = Transformer(
        src_vocab_size=src_vsize, tgt_vocab_size=tgt_vsize,
        d_model=d_model, N=N, num_heads=num_heads,
        d_ff=d_ff, dropout=dropout, pad_idx=PAD_IDX,
    ).to(device)
    model.src_vocab     = train_ds.src_vocab
    model.tgt_vocab     = train_ds.tgt_vocab
    model.src_tokenizer = train_ds.src_nlp
    model.tgt_tokenizer = train_ds.tgt_nlp

    if use_learned_pos_embed:
        class LearnedPositionalEmbedding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super().__init__()
                self.embed   = nn.Embedding(max_len, d_model)
                self.dropout = nn.Dropout(dropout)
            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
                return self.dropout(x + self.embed(pos))
        model.pos_enc = LearnedPositionalEmbedding(d_model, dropout).to(device)
        print("[train] Using learned positional embeddings")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable params: {n_params/1e6:.2f}M")

    if use_noam:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None

    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vsize, pad_idx=PAD_IDX, smoothing=label_smoothing)

    ema = EMA(model, decay=ema_decay) if use_ema else None
    if ema is not None:
        print(f"[train] EMA enabled (decay={ema_decay})")

    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float("inf")
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer=optimizer, scheduler=scheduler,
            epoch_num=epoch, is_train=True,
            device=device, pad_idx=PAD_IDX,
            grad_clip=grad_clip, wandb_run=wandb_run, ema=ema,
        )

        backup   = ema.copy_to(model) if ema is not None else None
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch, is_train=False,
            device=device, pad_idx=PAD_IDX,
        )

        weight_tag = "EMA" if ema is not None else "live"
        print(f"[epoch {epoch:02d}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  (eval on {weight_tag} weights)")
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train/epoch_loss": train_loss, "val/epoch_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                path=os.path.join(ckpt_dir, "best.pt"),
                src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab,
            )
            print(f"[train] new best val_loss={val_loss:.4f}, saved best.pt")
        else:
            epochs_since_improvement += 1
            print(f"[train] no improvement for {epochs_since_improvement} epoch(s)")

        if ema is not None:
            ema.restore(model, backup)

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            path=os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt"),
            src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab,
        )

        if epochs_since_improvement >= early_stop_patience:
            print(f"[train] early stopping triggered after {epoch+1} epochs")
            break

    print("[train] Computing test BLEU...")
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)
        model.to(device)

    bleu = evaluate_bleu(
        model, test_loader, train_ds.tgt_vocab, device=device,
        max_len=eval_max_len, use_beam=eval_use_beam,
        beam_size=eval_beam_size, length_penalty=eval_length_penalty,
    )
    mode = f"beam={eval_beam_size}, alpha={eval_length_penalty}" if eval_use_beam else "greedy"
    print(f"[train] Test BLEU ({mode}, max_len={eval_max_len}) = {bleu:.2f}")

    if wandb_run is not None:
        wandb_run.log({"test_bleu": bleu})
        wandb_run.finish()


if __name__ == "__main__":
    run_training_experiment()
