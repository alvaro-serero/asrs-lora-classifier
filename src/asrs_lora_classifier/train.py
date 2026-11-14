"""Train the LoRA classifier from a YAML config.

Run with: uv run python -m asrs_lora_classifier.train configs/debug.yaml
"""

import json
import random
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from asrs_lora_classifier.data import load_labels, load_split, make_loader
from asrs_lora_classifier.loader import REPO_ROOT
from asrs_lora_classifier.metrics import evaluate
from asrs_lora_classifier.model import build_classifier

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


def subsample(df, n, seed):
    """n random reports (all of them if n is None)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n, random_state=seed).reset_index(drop=True)


def predict_probs(clf, loader, device):
    clf.eval()
    probs, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = clf(b["input_ids"], b["attention_mask"])
            probs.append(torch.sigmoid(logits).cpu())
            ys.append(b["labels"].cpu())
            if device == "mps":
                torch.mps.empty_cache()
    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int)


def make_optimizer(clf, tcfg):
    lora_params = [p for n, p in clf.named_parameters() if p.requires_grad and not n.startswith("head")]
    return torch.optim.AdamW([
        {"params": lora_params, "lr": tcfg["lr_lora"]},
        {"params": clf.head.parameters(), "lr": tcfg["lr_head"]},
    ])


def save_checkpoint(clf, path):
    path.mkdir(parents=True, exist_ok=True)
    clf.body.save_pretrained(path)  # LoRA adapter only, not the frozen base
    torch.save(clf.head.state_dict(), path / "head.pt")


def main(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    mcfg, tcfg, run_name = cfg["model"], cfg["train"], cfg["run_name"]

    # Data
    labels = load_labels()
    mlb = MultiLabelBinarizer(classes=labels)
    full_train = load_split("train")
    label_freq = mlb.fit_transform(full_train["labels"]).mean(axis=0)  # from all of train
    train = subsample(full_train, cfg["data"]["train_subsample"], cfg["seed"])
    val = subsample(load_split("val"), cfg["data"]["val_subsample"], cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    train_loader = make_loader(train, mlb, tokenizer, mcfg["max_length"], tcfg["batch_size"], shuffle=True)
    val_loader = make_loader(val, mlb, tokenizer, mcfg["max_length"], tcfg["eval_batch_size"], shuffle=False)

    # Model, optimizer, schedule
    lcfg = cfg["lora"]
    clf = build_classifier(mcfg["name"], len(labels), device, label_freq,
                           dtype=getattr(torch, mcfg["dtype"]), r=lcfg["r"], lora_alpha=lcfg["alpha"],
                           lora_dropout=lcfg["dropout"], target_modules=lcfg["target_modules"])
    trainable = [p for p in clf.parameters() if p.requires_grad]
    optimizer = make_optimizer(clf, tcfg)

    accum = tcfg["grad_accum"]
    total_steps = (len(train_loader) // accum) * tcfg["epochs"]
    if tcfg["max_steps"] is not None:
        total_steps = min(total_steps, tcfg["max_steps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    run = None
    if cfg["wandb"]:
        import wandb
        run = wandb.init(project="asrs-lora-classifier", name=run_name, config=cfg)

    print(f"{device} | {len(train)} train, {len(val)} val | {total_steps} steps of "
          f"{tcfg['batch_size']} x {accum} reports")

    # Training loop
    step, best_f1, running_loss, train_time = 0, -1.0, 0.0, 0.0
    for epoch in range(tcfg["epochs"]):
        clf.train()
        optimizer.zero_grad()
        t0 = time.time()
        for i, b in enumerate(train_loader):
            b = b.to(device)
            logits = clf(b["input_ids"], b["attention_mask"])
            loss = loss_fn(logits, b["labels"]) / accum
            loss.backward()
            running_loss += loss.item()
            if device == "mps":
                torch.mps.empty_cache()
            if (i + 1) % accum != 0:
                continue

            # One optimizer step every `accum` batches
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if not np.isfinite(running_loss):
                raise RuntimeError(f"loss is {running_loss} at step {step}")

            if step % 10 == 0:
                train_time += time.time() - t0
                t0 = time.time()
                per_report = train_time / (step * tcfg["batch_size"] * accum)
                print(f"step {step}/{total_steps} | loss {running_loss / 10:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | {per_report * 1000:.0f} ms/report")
                if run:
                    run.log({"train/loss": running_loss / 10, "train/lr": scheduler.get_last_lr()[0],
                             "train/ms_per_report": per_report * 1000}, step=step)
                running_loss = 0.0

            if step % tcfg["eval_every"] == 0 or step == total_steps:
                train_time += time.time() - t0
                probs, y_val = predict_probs(clf, val_loader, device)
                res, thresholds = evaluate(probs, y_val)
                f1 = res["tuned"]["macro_f1"]
                print(f"  eval step {step} | macro-F1 @0.5 {res['at_0.5']['macro_f1']:.4f} | tuned {f1:.4f}")
                if run:
                    run.log({f"val/{k}/{m}": v for k, d in res.items() for m, v in d.items()}, step=step)

                if f1 > best_f1:
                    best_f1 = f1
                    save_checkpoint(clf, CHECKPOINT_DIR / run_name)
                    save_results(run_name, step, res, thresholds, probs, y_val, val, labels, cfg)
                    print(f"  new best, saved to checkpoints/{run_name}")
                clf.train()
                t0 = time.time()

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    print(f"best tuned macro-F1 on val: {best_f1:.4f}")
    if run:
        run.summary["best_val_macro_f1"] = best_f1
        run.finish()


def save_results(run_name, step, res, thresholds, probs, y_val, val, labels, cfg):
    """Same JSON format as tfidf_val.json, plus the val probabilities for error analysis."""
    RESULTS_DIR.mkdir(exist_ok=True)
    per_label = f1_score(y_val, (probs >= thresholds).astype(int), average=None, zero_division=0)
    out = {
        "model": run_name,
        "split": "val_2023",
        "best_step": step,
        "at_0.5": res["at_0.5"],
        "tuned_on_val": res["tuned"],
        "per_label_f1": dict(zip(labels, per_label.round(4).tolist())),
        "thresholds": dict(zip(labels, thresholds.round(2).tolist())),
        "config": cfg,
    }
    (RESULTS_DIR / f"{run_name}_val.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS_DIR / f"{run_name}_val_probs.npz",
             acn=np.array(val["acn"].tolist(), dtype=str), probs=probs)


if __name__ == "__main__":
    main(load_config(sys.argv[1]))
