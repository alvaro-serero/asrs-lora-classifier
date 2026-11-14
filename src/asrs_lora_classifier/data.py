"""Processed splits, label list, and the PyTorch Dataset / DataLoader for the LoRA model."""

import json
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from asrs_lora_classifier.loader import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "processed"


def load_split(name):
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def load_labels():
    return json.loads((DATA_DIR / "labels.json").read_text())


class ASRSDataset(Dataset):
    """One report = (narrative text, 0/1 label vector in the order of labels.json)."""

    def __init__(self, df, mlb):
        self.texts = df["narrative"].tolist()
        self.labels = torch.tensor(mlb.transform(df["labels"]), dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def collate(items, tokenizer, max_length):
    """Tokenize one batch, padding to the longest report in the batch."""
    texts, labels = zip(*items)
    batch = tokenizer(list(texts), padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    batch["labels"] = torch.stack(labels)
    return batch


def make_loader(df, mlb, tokenizer, max_length, batch_size, shuffle):
    return DataLoader(
        ASRSDataset(df, mlb), batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate, tokenizer=tokenizer, max_length=max_length),
    )
