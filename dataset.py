import json
import random
from collections import Counter
from pathlib import Path

import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']

DATA_DIR = Path(__file__).parent / "data"


class Vocab:
    # specials first (indices 0-3), then alphabetical for the rest
    def __init__(self, counter, min_freq=2, specials=SPECIALS):
        self.itos = list(specials)
        kept = sorted(
            tok for tok, freq in counter.items()
            if freq >= min_freq and tok not in specials
        )
        self.itos.extend(kept)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, tok):
        return self.stoi.get(tok, UNK_IDX)

    def __contains__(self, tok):
        return tok in self.stoi

    def get(self, tok, default=None):
        if default is None:
            default = UNK_IDX
        return self.stoi.get(tok, default)

    def lookup_token(self, idx):
        return self.itos[idx]


def _ensure_data_dir(data_dir: Path) -> None:
    # pull from HuggingFace if the local JSONL files are missing
    data_dir = Path(data_dir)
    needed = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
    missing = [name for name, fn in needed.items() if not (data_dir / fn).exists()]
    if not missing:
        return

    try:
        from datasets import load_dataset
    except ImportError:
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    hf_split = {"train": "train", "val": "validation", "test": "test"}
    for name in missing:
        try:
            ds = load_dataset("bentrevett/multi30k", split=hf_split[name])
        except Exception as e:
            print(f"[dataset] HuggingFace download failed for {name}: {e}")
            continue
        with open(data_dir / needed[name], "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps({"de": ex["de"], "en": ex["en"]}, ensure_ascii=False) + "\n")
        print(f"[dataset] wrote {data_dir / needed[name]} ({len(ds)} pairs)")


class Multi30kDataset(Dataset):
    # Multi30k DE->EN loaded from local JSONL files

    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        data_dir: Path = DATA_DIR,
        min_freq: int = 2,
        max_len: int = 100,
        auto_download: bool = True,
        word_dropout: float = 0.0,
    ):
        self.split = split
        self.max_len = max_len
        self.word_dropout = word_dropout

        if auto_download:
            _ensure_data_dir(Path(data_dir))

        candidates = [Path(data_dir) / f"{split}.jsonl"]
        if split == "validation":
            candidates.append(Path(data_dir) / "val.jsonl")
        elif split == "val":
            candidates.append(Path(data_dir) / "validation.jsonl")

        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Could not find {candidates[0]}. Place your JSONL files in "
                f"{data_dir}/ as train.jsonl, val.jsonl (or validation.jsonl), "
                f"test.jsonl."
            )

        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.pairs.append((ex["de"], ex["en"]))

        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")
        self.src_nlp = self.de_nlp
        self.tgt_nlp = self.en_nlp

        self.src_toks = [
            [t.text.lower() for t in self.de_nlp.tokenizer(s)]
            for s, _ in self.pairs
        ]
        self.tgt_toks = [
            [t.text.lower() for t in self.en_nlp.tokenizer(t)]
            for _, t in self.pairs
        ]

        # drop pairs that are too long to keep padding manageable
        keep = [
            i for i in range(len(self.pairs))
            if len(self.src_toks[i]) + 2 <= max_len
            and len(self.tgt_toks[i]) + 2 <= max_len
        ]
        if len(keep) < len(self.pairs):
            self.pairs    = [self.pairs[i]    for i in keep]
            self.src_toks = [self.src_toks[i] for i in keep]
            self.tgt_toks = [self.tgt_toks[i] for i in keep]

        if src_vocab is None or tgt_vocab is None:
            assert split == "train", (
                "val/test must receive the train vocabularies via "
                "src_vocab=, tgt_vocab="
            )
            src_counter = Counter(tok for sent in self.src_toks for tok in sent)
            tgt_counter = Counter(tok for sent in self.tgt_toks for tok in sent)
            self.src_vocab = Vocab(src_counter, min_freq=min_freq)
            self.tgt_vocab = Vocab(tgt_counter, min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src_ids = [SOS_IDX] + [self.src_vocab[t] for t in self.src_toks[i]] + [EOS_IDX]
        tgt_ids = [SOS_IDX] + [self.tgt_vocab[t] for t in self.tgt_toks[i]] + [EOS_IDX]

        # word dropout for train only; off by default (0.0)
        if self.word_dropout > 0.0 and self.split == "train":
            src_ids = [
                UNK_IDX if (idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                            and random.random() < self.word_dropout)
                else idx
                for idx in src_ids
            ]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

    def src_len(self, i: int) -> int:
        return len(self.src_toks[i]) + 2


def collate_fn(batch, pad_idx=PAD_IDX):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=pad_idx)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_idx)
    return src, tgt
