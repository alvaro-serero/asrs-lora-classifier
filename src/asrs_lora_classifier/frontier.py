"""Prompted frontier-model baseline (Claude, few-shot, no training) on the 1,000-report test sample.

Try first:  uv run python -m asrs_lora_classifier.frontier --limit 20
Full run:   uv run python -m asrs_lora_classifier.frontier
Needs ANTHROPIC_API_KEY. Every answer is appended to results/frontier_raw.jsonl as it arrives,
so a rerun only calls the API for reports that are not done yet.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from asrs_lora_classifier.data import load_baseline_acns, load_labels, load_split
from asrs_lora_classifier.loader import REPO_ROOT
from asrs_lora_classifier.metrics import scores

RESULTS_DIR = REPO_ROOT / "results"
RAW_PATH = RESULTS_DIR / "frontier_raw.jsonl"

MODEL = "claude-opus-5-5"
# USD per million tokens (platform.claude.com/docs/en/about-claude/pricing, Nov 2026)
PRICE = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_write": 5.0, "cache_read": 0.20}
N_EXAMPLES = 5


def few_shot_examples(n, seed=42):
    """n short train reports from 2020+ (after the taxonomy change), with their labels."""
    train = load_split("train")
    pool = train[(train["year"] >= 2020) & (train["narrative"].str.split().str.len() <= 200)]
    return pool.sample(n, random_state=seed)


def build_system(labels, examples):
    label_list = "\n".join(f"- {label}" for label in labels)
    shots = "\n\n".join(
        f"<example>\n<narrative>\n{r.narrative}\n</narrative>\n<labels>{json.dumps(list(r.labels))}</labels>\n</example>"
        for r in examples.itertuples()
    )
    return (
        "You classify NASA ASRS aviation incident reports. Given the narrative written by the reporter, "
        "return every anomaly label from the list below that applies to the event. A report usually has "
        "1 to 4 labels. Use only labels from the list, spelled exactly as written.\n\n"
        f"<labels>\n{label_list}\n</labels>\n\nLabeled examples:\n\n{shots}"
    )


class BadAnswer(Exception):
    pass


def make_schema(labels):
    """JSON schema for the answer: one list of labels, restricted to the 46 names by the enum."""
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": labels}}},
            "required": ["labels"],
            "additionalProperties": False,
        },
    }


def classify(client, system, schema, canonical, narrative):
    t0 = time.time()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,  # Opus thinks before answering, and thinking counts toward max_tokens
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"format": schema},  # structured output: the reply must match the schema
        messages=[{"role": "user", "content": f"<narrative>\n{narrative}\n</narrative>"}],
    )
    if msg.stop_reason != "end_turn":
        raise BadAnswer(f"stop_reason={msg.stop_reason}, output_tokens={msg.usage.output_tokens}")
    text = next(b.text for b in msg.content if b.type == "text")
    raw = json.loads(text)["labels"]
    # The model may change the capitalization of enum values: map back to the exact label names
    labels = [canonical[x.lower()] for x in raw if x.lower() in canonical]
    u = msg.usage
    return {
        "labels": labels,
        "n_dropped": len(raw) - len(labels),
        "latency_s": round(time.time() - t0, 2),
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_write": u.cache_creation_input_tokens or 0,
        "cache_read": u.cache_read_input_tokens or 0,
    }


def main(limit, workers):
    labels = load_labels()
    acns = load_baseline_acns()
    test = load_split("test")
    sample = test[test["acn"].isin(set(acns))].set_index("acn").loc[acns].reset_index()
    if limit:
        sample = sample.head(limit)

    done = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text().splitlines():
            rec = json.loads(line)
            done[rec["acn"]] = rec
    todo = [r for r in sample.itertuples() if r.acn not in done]

    system, schema = build_system(labels, few_shot_examples(N_EXAMPLES)), make_schema(labels)
    canonical = {label.lower(): label for label in labels}
    client = anthropic.Anthropic(max_retries=8)
    print(f"{MODEL} | {len(sample)} reports, {len(sample) - len(todo)} already done, {len(todo)} to call")

    def run(row):
        """One report. A bad answer is returned as an error instead of stopping the whole run."""
        try:
            return {"acn": row.acn, **classify(client, system, schema, canonical, row.narrative)}
        except (BadAnswer, json.JSONDecodeError, KeyError, StopIteration) as e:
            return {"acn": row.acn, "error": f"{type(e).__name__}: {e}"}

    RESULTS_DIR.mkdir(exist_ok=True)
    errors = []
    with RAW_PATH.open("a") as f:
        def save(rec):
            if "error" in rec:  # not saved, so a rerun tries this report again
                errors.append(rec)
                return
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[rec["acn"]] = rec

        if todo:
            save(run(todo[0]))  # first call alone writes the prompt cache; the rest read it
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(run, r) for r in todo[1:]]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                save(fut.result())

    if errors:
        for e in errors:
            print(e["acn"], e["error"])
        print(f"{len(errors)} reports failed and were not saved: rerun the same command to retry them.")
        return

    # Score: the model returns label sets, so there are no thresholds to tune
    recs = [done[a] for a in sample["acn"]]
    mlb = MultiLabelBinarizer(classes=labels).fit([labels])
    y, y_pred = mlb.transform(sample["labels"]), mlb.transform([r["labels"] for r in recs])
    per_label = f1_score(y, y_pred, average=None, zero_division=0)

    usage = {k: sum(r[k] for r in recs) for k in PRICE}
    cost = sum(usage[k] * PRICE[k] for k in PRICE) / 1e6
    out = {
        "model": MODEL,
        "n_examples": N_EXAMPLES,
        "baseline_sample": {
            "n_reports": len(sample),
            "scores": scores(y, y_pred),
            "per_label_f1": dict(zip(labels, per_label.round(4).tolist())),
        },
        "usage": usage,
        "cost_usd": round(cost, 3),
        "cost_per_1000_reports_usd": round(cost / len(sample) * 1000, 2),
        "labels_dropped": sum(r["n_dropped"] for r in recs),
        "median_latency_s": float(np.median([r["latency_s"] for r in recs])),
    }
    (RESULTS_DIR / "frontier_test.json").write_text(json.dumps(out, indent=2))
    print("scores:", {k: round(v, 4) for k, v in out["baseline_sample"]["scores"].items()})
    print(f"cost ${cost:.2f} (${out['cost_per_1000_reports_usd']} per 1,000) | "
          f"median latency {out['median_latency_s']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N sample reports")
    parser.add_argument("--workers", type=int, default=8, help="parallel API calls")
    args = parser.parse_args()
    main(args.limit, args.workers)
