import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except ImportError:
    gdown = None


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # safe for fully-masked rows

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # [batch, 1, 1, src_len] — True where token is PAD
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # padding mask OR'd with causal mask, shape [batch, 1, tgt_len, tgt_len]
    _, tgt_len = tgt.size()
    device = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = self.dropout(attn_out)
        self.last_attn_weights = attn_w.detach()

        return self.W_o(self._merge_heads(attn_out))


class PositionalEncoding(nn.Module):
    # sinusoidal PE, stored as a non-trainable buffer

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class PositionwiseFeedForward(nn.Module):
    # FFN(x) = max(0, xW1 + b1)W2 + b2

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    # Pre-LN: norm before attention/FFN for stable training

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        nx = self.norm1(x)
        x = x + self.dropout1(self.self_attn(nx, nx, nx, src_mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        nx = self.norm1(x)
        x = x + self.dropout1(self.self_attn(nx, nx, nx, tgt_mask))
        nx = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(nx, memory, memory, src_mask))
        x = x + self.dropout3(self.ff(self.norm3(x)))
        return x


class Encoder(nn.Module):

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    # base config: d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1
    # tgt_embed and generator share weights (paper §3.4)

    # https://drive.google.com/file/d/1JY9FKdtwOsUVRi23L3Ccf4a5nRUTkYrT/view?usp=sharing
    DEFAULT_DRIVE_ID = "1JY9FKdtwOsUVRi23L3Ccf4a5nRUTkYrT"
    DEFAULT_CKPT_PATH = "best.pt"

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        pad_idx: int = 1,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()

        ckpt = None
        if src_vocab_size is None or tgt_vocab_size is None:
            path = checkpoint_path or self.DEFAULT_CKPT_PATH

            if not os.path.exists(path) and gdown is not None:
                try:
                    gdown.download(id=self.DEFAULT_DRIVE_ID, output=path, quiet=False)
                except Exception as e:
                    print(f"[Transformer] gdown download failed: {e}")

            if os.path.exists(path):
                ckpt = self._safe_load(path)
                cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
                src_vocab_size = src_vocab_size or cfg.get("src_vocab_size")
                tgt_vocab_size = tgt_vocab_size or cfg.get("tgt_vocab_size")
                d_model    = cfg.get("d_model", d_model)
                N          = cfg.get("N", N)
                num_heads  = cfg.get("num_heads", num_heads)
                d_ff       = cfg.get("d_ff", d_ff)
                dropout    = cfg.get("dropout", dropout)
                pad_idx    = cfg.get("pad_idx", pad_idx)
                tie_weights = cfg.get("tie_weights", tie_weights)

            if src_vocab_size is None or tgt_vocab_size is None:
                raise RuntimeError(
                    f"Transformer() called with no vocab sizes and no usable "
                    f"checkpoint at '{path}'. Either provide vocab sizes or "
                    f"place a valid checkpoint at this path."
                )

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model   = d_model
        self.N         = N
        self.num_heads = num_heads
        self.d_ff      = d_ff
        self.dropout_p = dropout
        self.pad_idx   = pad_idx
        self.tie_weights = tie_weights

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # tie tgt embedding and output projection to save params
        if self.tie_weights:
            self._tie_embeddings_and_generator()

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None

        if ckpt is not None:
            self._restore_from_ckpt(ckpt)
        elif checkpoint_path is not None:
            self._maybe_download_and_load(checkpoint_path)

        if self.src_tokenizer is None:
            try:
                import spacy
                self.src_tokenizer = spacy.load("de_core_news_sm")
            except Exception as e:
                print(f"[Transformer] could not load spaCy de_core_news_sm: {e}")

    def _tie_embeddings_and_generator(self) -> None:
        self.generator.weight = self.tgt_embed.weight

    @staticmethod
    def _safe_load(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _restore_from_ckpt(self, ckpt) -> None:
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.src_vocab = ckpt.get("src_vocab", None)
            self.tgt_vocab = ckpt.get("tgt_vocab", None)
        else:
            self.load_state_dict(ckpt, strict=False)
        # re-tie after loading because load creates fresh tensors
        if self.tie_weights:
            self._tie_embeddings_and_generator()

    def _maybe_download_and_load(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path) and gdown is not None:
            try:
                gdown.download(id=self.DEFAULT_DRIVE_ID, output=checkpoint_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] gdown download failed: {e}")

        if os.path.exists(checkpoint_path):
            ckpt = self._safe_load(checkpoint_path)
            self._restore_from_ckpt(ckpt)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        return self.encoder(self.pos_enc(x), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.decoder(self.pos_enc(x), memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(self, src, tgt, src_mask, tgt_mask) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def get_config(self) -> dict:
        return dict(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            N=self.N,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout_p,
            pad_idx=self.pad_idx,
            tie_weights=self.tie_weights,
        )

    @staticmethod
    def _vocab_get(vocab, key, default=None):
        if vocab is None:
            return default
        if hasattr(vocab, "get") and callable(getattr(vocab, "get")):
            try:
                return vocab.get(key, default)
            except TypeError:
                pass
        try:
            return vocab[key]
        except Exception:
            return default

    @staticmethod
    def _vocab_itos(vocab) -> dict:
        if vocab is None:
            return {}
        if hasattr(vocab, "itos"):
            return {i: t for i, t in enumerate(vocab.itos)}
        if hasattr(vocab, "get_itos"):
            return {i: t for i, t in enumerate(vocab.get_itos())}
        if isinstance(vocab, dict):
            return {idx: tok for tok, idx in vocab.items()}
        return {}

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        # greedy decode a German sentence to English
        assert self.src_vocab is not None and self.tgt_vocab is not None, (
            "Vocabs not set on the model. Construct with checkpoint_path= or "
            "rely on no-arg construction (auto-downloads checkpoint)."
        )

        from train import greedy_decode

        device = next(self.parameters()).device
        self.eval()

        if self.src_tokenizer is not None:
            tokens = [t.text.lower() for t in self.src_tokenizer.tokenizer(src_sentence)]
        else:
            tokens = src_sentence.lower().strip().split()

        sv, tv = self.src_vocab, self.tgt_vocab
        s_unk = self._vocab_get(sv, "<unk>", 0)
        s_sos = self._vocab_get(sv, "<sos>", 2)
        s_eos = self._vocab_get(sv, "<eos>", 3)
        s_pad = self._vocab_get(sv, "<pad>", 1)
        t_sos = self._vocab_get(tv, "<sos>", 2)
        t_eos = self._vocab_get(tv, "<eos>", 3)

        src_ids = [s_sos] + [self._vocab_get(sv, tok, s_unk) for tok in tokens] + [s_eos]
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=s_pad)

        out_ids = greedy_decode(
            self, src, src_mask,
            max_len=max_len,
            start_symbol=t_sos,
            end_symbol=t_eos,
            device=device,
        )

        itos = self._vocab_itos(tv)
        out_tokens = []
        for idx in out_ids[0].cpu().tolist():
            if idx == t_sos:
                continue
            if idx == t_eos:
                break
            tok = itos.get(idx, "<unk>")
            if tok in ("<unk>", "<pad>", "<sos>", "<eos>"):
                continue
            out_tokens.append(tok)
        return " ".join(out_tokens)
