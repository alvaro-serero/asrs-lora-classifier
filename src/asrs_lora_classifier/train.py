"""Train the LoRA classifier from a YAML config.

Run with: uv run python -m asrs_lora_classifier.train configs/debug.yaml
"""

import random
import sys

import numpy as np
import torch
import yaml

from asrs_lora_classifier.loader import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


# Part 2: predict_probs, train loop with warmup + accumulation + eval every N steps.


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    print(cfg)
