import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    use_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    # use_scale=False skips the 1/sqrt(dk) term (W&B Exp 2 ablation)
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)  # safe for fully-masked rows
    return torch.matmul(attn_w, V), attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # [B, 1, 1, T] — True where PAD
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # padding mask OR'd with causal mask, [B, 1, T, T]
    B, T   = tgt.shape
    pad_m  = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    return pad_m | causal


class MultiHeadAttention(nn.Module):
    # attn_weights cached after each forward for W&B attention visualisation

    def __init__(self, d_model, num_heads, dropout=0.1, *, use_scale=True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        self.W_q     = nn.Linear(d_model, d_model)
        self.W_k     = nn.Linear(d_model, d_model)
        self.W_v     = nn.Linear(d_model, d_model)
        self.W_o     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, query, key, value, mask=None) -> torch.Tensor:
        B = query.size(0)

        def _split(linear, x):
            return linear(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = _split(self.W_q, query)
        K = _split(self.W_k, key)
        V = _split(self.W_v, value)

        out, attn_w = scaled_dot_product_attention(Q, K, V, mask, use_scale=self.use_scale)
        self.attn_weights = attn_w.detach()  # keep for heatmaps

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class PositionalEncoding(nn.Module):
    # sinusoidal PE stored as a buffer (not a trainable param)

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class LearnedPositionalEncoding(nn.Module):
    # learned PE via nn.Embedding; drop-in replacement for W&B Exp 4

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(pos))


class PositionwiseFeedForward(nn.Module):
    # FFN(x) = max(0, xW1 + b1)W2 + b2

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    # Post-LN: matches the original paper spec

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, *, use_scale=True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, *, use_scale=True) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, memory, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    # full encoder-decoder transformer
    # pe_type and use_scale are W&B ablation flags

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
        pad_idx:        int   = 1,
        pe_type:        str   = 'sinusoidal',
        use_scale:      bool  = True,
        src_vocab              = None,
        tgt_vocab              = None,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.pad_idx   = pad_idx
        self.pe_type   = pe_type
        self.use_scale = use_scale
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self._de_nlp   = None  # lazy-loaded on first infer() call

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        PE = PositionalEncoding if pe_type == 'sinusoidal' else LearnedPositionalEncoding
        self.src_pe = PE(d_model, dropout)
        self.tgt_pe = PE(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale=use_scale)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale=use_scale)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(ckpt['model_state_dict'])
            if 'src_vocab' in ckpt and self.src_vocab is None:
                self.src_vocab = ckpt['src_vocab']
            if 'tgt_vocab' in ckpt and self.tgt_vocab is None:
                self.tgt_vocab = ckpt['tgt_vocab']

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pe(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.fc_out(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        # greedy decode a German sentence to English
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Attach src_vocab and tgt_vocab before calling infer().\n"
                "They are saved/restored automatically by save/load_checkpoint."
            )
        if self._de_nlp is None:
            import spacy
            self._de_nlp = spacy.load("de_core_news_sm")

        device   = next(self.parameters()).device
        SOS, EOS = 2, 3

        tokens   = [t.text.lower() for t in self._de_nlp.tokenizer(src_sentence)]
        src_ids  = [SOS] + [self.src_vocab[t] for t in tokens] + [EOS]
        src      = torch.tensor([src_ids], device=device)
        src_mask = make_src_mask(src, self.pad_idx)

        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys     = torch.tensor([[SOS]], device=device)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == EOS:
                    break

        skip = {'<eos>', '<sos>', '<pad>'}
        out  = []
        for idx in ys[0, 1:]:
            tok = self.tgt_vocab.lookup_token(idx.item())
            if tok == '<eos>':
                break
            if tok not in skip:
                out.append(tok)
        return ' '.join(out)
