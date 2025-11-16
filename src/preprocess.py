"""src/preprocess.py – complete data-loading / preprocessing"""
from __future__ import annotations

import random
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as T
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import CIFAR10

from datasets import load_dataset  # 🤗 datasets – for IWSLT14
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

CACHE_DIR = Path(get_original_cwd()) / ".cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CIFAR-10 loaders
# ---------------------------------------------------------------------------

def _cifar10(cfg: DictConfig, seed: int):
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    aug = []
    if cfg.dataset.augmentations.random_crop:
        aug.append(T.RandomCrop(cfg.dataset.image_size, padding=4))
    if cfg.dataset.augmentations.random_flip:
        aug.append(T.RandomHorizontalFlip())
    train_tf = T.Compose(aug + [T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])

    full_train = CIFAR10(str(CACHE_DIR), train=True, download=True, transform=train_tf)
    test_set = CIFAR10(str(CACHE_DIR), train=False, download=True, transform=test_tf)

    # 5k-sample validation
    idxs = list(range(len(full_train)))
    random.Random(seed).shuffle(idxs)
    val_idx, train_idx = idxs[:5000], idxs[5000:]
    train_set = data.Subset(full_train, train_idx)
    val_set = data.Subset(CIFAR10(str(CACHE_DIR), train=True, transform=test_tf), val_idx)

    dl_args = dict(batch_size=cfg.dataset.batch_size, num_workers=2, pin_memory=True)
    return (
        data.DataLoader(train_set, shuffle=True, **dl_args),
        data.DataLoader(val_set, shuffle=False, **dl_args),
        data.DataLoader(test_set, shuffle=False, **dl_args),
        {"num_classes": cfg.dataset.num_classes, "task": "clf"},
    )


# ---------------------------------------------------------------------------
# IWSLT14 De→En loaders (using 🤗 datasets)  ----------------------------------
# ---------------------------------------------------------------------------
TOKEN_SRC = get_tokenizer("basic_english")
TOKEN_TGT = get_tokenizer("basic_english")
SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK, PAD, BOS, EOS = range(4)


def _build_vocab(examples, side: str, vocab_size: int) -> Vocab:
    counter = Counter()
    for ex in examples:
        text = ex["de"] if side == "src" else ex["en"]
        tokens = TOKEN_SRC(text) if side == "src" else TOKEN_TGT(text)
        counter.update(tokens)
    return Vocab(counter, max_size=vocab_size, specials=SPECIALS, min_freq=2)


def _numberise(tokens: List[str], vocab: Vocab):
    return [BOS] + [vocab[t] for t in tokens] + [EOS]


def _collate_fn_factory(src_vocab: Vocab, tgt_vocab: Vocab, max_len: int):
    def _collate(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_tok = TOKEN_SRC(item["de"])[:max_len]
            tgt_tok = TOKEN_TGT(item["en"])[:max_len]
            src_batch.append(torch.tensor(_numberise(src_tok, src_vocab)))
            tgt_batch.append(torch.tensor(_numberise(tgt_tok, tgt_vocab)))
        src_pad = pad_sequence(src_batch, padding_value=PAD)
        tgt_pad = pad_sequence(tgt_batch, padding_value=PAD)
        return src_pad, tgt_pad
    return _collate


def _iwslt14(cfg: DictConfig, seed: int):
    ds = load_dataset("iwslt2017", "de-en", split={"train": "train", "valid": "validation", "test": "test"}, cache_dir=str(CACHE_DIR))
    train_examples = list(ds["train"])
    random.Random(seed).shuffle(train_examples)

    src_vocab = _build_vocab(train_examples, "src", cfg.dataset.bpe_vocab_size)
    tgt_vocab = _build_vocab(train_examples, "tgt", cfg.dataset.bpe_vocab_size)

    collate = _collate_fn_factory(src_vocab, tgt_vocab, cfg.dataset.max_seq_length)

    dl_args = dict(batch_size=None, num_workers=2)
    return (
        data.DataLoader(train_examples, shuffle=True, collate_fn=collate, **dl_args),
        data.DataLoader(ds["valid"], shuffle=False, collate_fn=collate, **dl_args),
        data.DataLoader(ds["test"], shuffle=False, collate_fn=collate, **dl_args),
        {
            "task": "nmt",
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "pad_idx": PAD,
            "bos_idx": BOS,
            "eos_idx": EOS,
        },
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def get_data_loaders(cfg: DictConfig, seed: int):
    if cfg.dataset.name == "cifar10":
        return _cifar10(cfg, seed)
    if cfg.dataset.name.startswith("iwslt14"):
        return _iwslt14(cfg, seed)
    raise ValueError(cfg.dataset.name)