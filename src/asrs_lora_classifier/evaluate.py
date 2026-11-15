"""Score a trained LoRA checkpoint on the 2024-25 test set, with the per-label thresholds tuned on val.

Run with: uv run python -m asrs_lora_classifier.evaluate configs/kaggle.yaml
Needs checkpoints/<run_name>/ and results/<run_name>_val.json (both written by train.py).
"""

import json
import sys

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoTokenizer

from asrs_lora_classifier.data import load_baseline_acns, load_labels, load_split, make_loader
from asrs_lora_classifier.metrics import test_report
from asrs_lora_classifier.model import load_classifier
from asrs_lora_classifier.train import CHECKPOINT_DIR, RESULTS_DIR, get_device, load_config, predict_probs


def main(cfg):
    device = get_device(cfg["device"])
    mcfg, run_name = cfg["model"], cfg["run_name"]

    labels = load_labels()
    mlb = MultiLabelBinarizer(classes=labels).fit([labels])
    test = load_split("test").reset_index(drop=True)

    # Thresholds come from val only: test is never used to choose anything
    val_res = json.loads((RESULTS_DIR / f"{run_name}_val.json").read_text())
    thresholds = np.array([val_res["thresholds"][label] for label in labels])

    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    loader = make_loader(test, mlb, tokenizer, mcfg["max_length"], cfg["train"]["eval_batch_size"], shuffle=False)
    clf = load_classifier(mcfg["name"], CHECKPOINT_DIR / run_name, len(labels), device,
                          dtype=getattr(torch, mcfg["dtype"]))

    print(f"{device} | {len(test)} test reports | checkpoint from val step {val_res['best_step']}")
    probs, y_test = predict_probs(clf, tqdm(loader), device)

    report = test_report(test, y_test, probs, thresholds, labels, load_baseline_acns())
    out = {"model": run_name, "best_val_step": val_res["best_step"], **report}
    (RESULTS_DIR / f"{run_name}_test.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS_DIR / f"{run_name}_test_probs.npz",
             acn=np.array(test["acn"].tolist(), dtype=str), probs=probs)

    for part in ("full_test", "baseline_sample"):
        print(part, {k: round(v, 4) for k, v in report[part]["val_thresholds"].items()})


if __name__ == "__main__":
    main(load_config(sys.argv[1]))
