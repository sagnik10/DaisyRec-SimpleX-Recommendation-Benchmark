#!/usr/bin/env python3
"""
Plot tune run of every (algorithm, dataset) into one figure.

Usage:
    python3 plots_all.py <log_root> [output_name]

    <log_root> is the folder containing one sub-folder per algorithm
    (log/mf, log/neumf, log/simplex, ...), Example.  python3 plots_all.py log

"""
import os, re, sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

ALGOS    = ["mf", "neumf", "simplex"]
DATASETS = ["ml-50k", "yelp-200k"]

BEST_RE  = re.compile(r"Trial\s+(\d+)\s+get the best ndcg\(([0-9.eE+-]+)\)")
TRIAL_RE = re.compile(r"Finish\s+(\d+)\s+trial")
DATA_RE  = re.compile(r"Current data path is:\s*data/([^/\s]+)/")
TIME_RE  = re.compile(r"^\w{3}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$")


def parse_tune_log(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    best = BEST_RE.findall(text)
    if not best:
        return None
    ds = DATA_RE.search(text)
    n_trials = [int(x) for x in TRIAL_RE.findall(text)]
    stem = os.path.splitext(os.path.basename(path))[0]
    when = datetime.strptime(stem, "%b-%d-%Y_%H-%M-%S") if TIME_RE.match(stem) \
        else datetime.fromtimestamp(os.path.getmtime(path))
    return dict(path=path, time=when,
                dataset=ds.group(1) if ds else "unknown",
                best_trial=int(best[-1][0]), best_ndcg=float(best[-1][1]),
                n_trials=max(n_trials) if n_trials else None)


def latest_tunes(root):
    latest = {}
    for algo in ALGOS:
        d = os.path.join(root, algo)
        if not os.path.isdir(d):
            print(f"[warn] missing folder {d}")
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".log"):
                continue
            r = parse_tune_log(os.path.join(d, fn))
            if r and r["dataset"] in DATASETS:
                key = (algo, r["dataset"])
                if key not in latest or r["time"] > latest[key]["time"]:
                    latest[key] = r
    return latest


def plot(latest, name, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    colors = {"mf": "#4C72B0", "neumf": "#DD8452", "simplex": "#55A868"}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 5))
    for ax, ds in zip(axes, DATASETS):
        vals = [latest.get((a, ds)) for a in ALGOS]
        x = np.arange(len(ALGOS))
        h = [v["best_ndcg"] if v else 0 for v in vals]
        ax.bar(x, h, color=[colors[a] for a in ALGOS], width=0.6)
        for xi, v in zip(x, vals):
            if v:
                ax.text(xi, v["best_ndcg"], f"{v['best_ndcg']:.4f}\n(trial {v['best_trial']}/{v['n_trials']})",
                        ha="center", va="bottom", fontsize=9)
            else:
                ax.text(xi, 0, "no tune", ha="center", va="bottom", fontsize=9, color="gray")
        ax.set_xticks(x); ax.set_xticklabels([a.upper() if a != "neumf" else "NeuMF" for a in ALGOS])
        ax.set_title(ds); ax.set_ylabel("Best validation NDCG")
        ax.set_ylim(0, max(h) * 1.25 if max(h) else 1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Hyperparameter tuning – best NDCG of latest tune", fontsize=13)
    fig.tight_layout()
    out = os.path.join(save_dir, f"{name}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved: {out}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    root = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "tuning_comparison"
    latest = latest_tunes(root)
    print(f"{'algo':8} {'dataset':10} {'file':28} {'best trial':>10} {'trials':>6} {'NDCG':>9}")
    for (a, d), r in sorted(latest.items()):
        print(f"{a:8} {d:10} {os.path.basename(r['path']):28} {r['best_trial']:>10} {str(r['n_trials']):>6} {r['best_ndcg']:>9.4f}")
    plot(latest, name)


if __name__ == "__main__":
    main()