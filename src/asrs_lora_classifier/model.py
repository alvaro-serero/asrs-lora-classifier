"""Qwen3 body with LoRA adapters + last-token pooling + a linear multi-label head."""

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel


class ASRSClassifier(nn.Module):
    def __init__(self, body, hidden_size, n_labels):
        super().__init__()
        self.body = body
        self.head = nn.Linear(hidden_size, n_labels)  # float32

    def forward(self, input_ids, attention_mask):
        h = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1  # right padding: last real token
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
        return self.head(pooled.float())  # logits, shape (batch, n_labels)


def build_classifier(model_name, n_labels, device, label_freq=None, dtype=torch.bfloat16,
                     r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    body = AutoModel.from_pretrained(model_name, dtype=dtype)
    hidden_size = body.config.hidden_size
    lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=list(target_modules))
    body = get_peft_model(body, lora_config)
    clf = ASRSClassifier(body, hidden_size, n_labels)
    if label_freq is not None:
        # Head starts by predicting each label's train frequency (see notebook 7.4).
        nn.init.zeros_(clf.head.weight)
        with torch.no_grad():
            clf.head.bias.copy_(torch.tensor(np.log(label_freq / (1 - label_freq)), dtype=torch.float32))
    return clf.to(device)


def load_classifier(model_name, path, n_labels, device, dtype=torch.bfloat16):
    """Rebuild a trained classifier from a folder written by train.save_checkpoint, for inference."""
    body = AutoModel.from_pretrained(model_name, dtype=dtype)
    hidden_size = body.config.hidden_size
    body = PeftModel.from_pretrained(body, path)  # attaches the saved LoRA adapter (frozen)
    clf = ASRSClassifier(body, hidden_size, n_labels)
    clf.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
    return clf.to(device).eval()
