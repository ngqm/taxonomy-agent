"""Pluggable classifiers for the cascade: label the confident majority of a
corpus from a judge-labeled calibration set (the discovery probes plus a fresh
re-judge against the final taxonomy).

Two options:

- ``"prototype"`` — nearest class-mean on frozen MiniLM embeddings (no training;
  the "embedding clustering" cascade; falls back to the category description for
  classes with no training example).
- ``"finetune"`` — fine-tune a BERT-family encoder end to end (heavier; benefits
  most from a larger re-judged calibration set; predicts only trained classes).

Each exposes ``fit(texts, labels)`` then ``predict(text_iter) -> (labels,
confidence)``, where ``confidence`` is "higher = more sure" so the cascade gate
keeps the top ``coverage`` fraction and routes the rest to the judge. Heavy deps
(torch, transformers) import lazily, so choosing one classifier never forces the
other.
"""
from __future__ import annotations

import numpy as np


DEFAULT_FINETUNE_MODEL = "distilbert-base-uncased"


def _batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


class PrototypeClassifier:
    """Nearest class-mean on frozen embeddings — no training."""

    def __init__(self, embed_fn, targets, descriptions=None):
        self.embed_fn = embed_fn
        self.targets = list(targets)
        self.descriptions = descriptions
        self.names, self.mat = [], None

    def fit(self, texts, labels):
        from . import cascade
        self.names, self.mat = cascade.build_prototypes(
            list(zip(texts, labels)), self.targets, self.embed_fn,
            self.descriptions)

    def predict(self, text_iter, batch_size=1024):
        from . import cascade
        return cascade.assign_streaming(
            self.names, self.mat, text_iter, self.embed_fn, batch_size)


class FinetuneClassifier:
    """Fine-tune a BERT-family encoder end to end on the calibration set."""

    def __init__(self, base_model=DEFAULT_FINETUNE_MODEL, epochs=4, lr=5e-5,
                 batch_size=16, max_len=256, seed=42):
        self.base_model = base_model
        self.epochs = epochs
        self.lr = lr
        self.train_bs = batch_size
        self.max_len = max_len
        self.seed = seed
        self.labels_ = []

    def fit(self, texts, labels):
        import random
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        texts, labels = list(texts), list(labels)
        self.labels_ = sorted(set(labels))
        if len(self.labels_) < 2:
            return                    # single class → predict() returns it constant
        l2i = {lbl: i for i, lbl in enumerate(self.labels_)}
        torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=len(self.labels_)).to(self.device)
        y = torch.tensor([l2i[lbl] for lbl in labels])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        rng = random.Random(self.seed)
        order = list(range(len(texts)))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for s in range(0, len(order), self.train_bs):
                bi = order[s:s + self.train_bs]
                enc = self.tok([texts[i] for i in bi], truncation=True,
                               padding=True, max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                out = self.model(**enc, labels=y[bi].to(self.device))
                out.loss.backward()
                opt.step()
                opt.zero_grad()
        self.model.eval()

    def predict(self, text_iter, batch_size=None):
        import torch
        bs = batch_size or 64
        if len(self.labels_) < 2:
            const = self.labels_[0] if self.labels_ else "other"
            preds, n = [], 0
            for batch in _batched(text_iter, bs):
                preds += [const] * len(batch)
                n += len(batch)
            return preds, np.ones(n, dtype=np.float32)
        preds, conf = [], []
        with torch.no_grad():
            for batch in _batched(text_iter, bs):
                enc = self.tok(list(batch), truncation=True, padding=True,
                               max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                proba = torch.softmax(self.model(**enc).logits, dim=-1)
                p, idx = proba.max(dim=-1)
                preds += [self.labels_[j] for j in idx.tolist()]
                conf.append(p.cpu().numpy().astype(np.float32))
        return preds, (np.concatenate(conf) if conf
                       else np.zeros(0, dtype=np.float32))


def make_classifier(kind, *, embed_fn=None, targets=None, descriptions=None,
                    finetune_model=DEFAULT_FINETUNE_MODEL, epochs=4, seed=42):
    """Construct the cascade classifier. ``prototype`` needs ``embed_fn``;
    ``finetune`` tokenizes raw text and ignores it."""
    if kind == "prototype":
        return PrototypeClassifier(embed_fn, targets or [], descriptions)
    if kind == "finetune":
        return FinetuneClassifier(finetune_model, epochs=epochs, seed=seed)
    raise ValueError(f"unknown classifier {kind!r} "
                     f"(expected 'prototype' or 'finetune')")
