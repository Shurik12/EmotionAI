#!/usr/bin/env python3
"""
Measure precision and recall of the audio burnout model via the server API.

Usage:
    # Labelled evaluation on training set
    python3 tests/python/measure_burnout.py --train tests/data/burnout/train \\
        --train-targets tests/data/burnout/train_targets.tsv --output results.csv

    # Unlabelled distribution on control/burnout sets
    python3 tests/python/measure_burnout.py
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_API_BASE = os.environ.get("VITE_API_URL", "https://razuma.tech/api")
UPLOAD_ENDPOINT = "/upload_burnout"
PROGRESS_ENDPOINT = "/progress/{task_id}"
MODEL_NAME = "emotieff"

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 600

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".mp4", ".avi", ".webm"}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONTROL_DIR = REPO_ROOT / "tests" / "data" / "control"
DEFAULT_BURNOUT_DIR = REPO_ROOT / "tests" / "data" / "burnout" / "test"

LEVEL_ORDER = ["low", "moderate", "high", "severe"]


# ============================================================
# API helpers
# ============================================================
def upload_audio(api_base: str, file_path: Path) -> dict:
    url = f"{api_base.rstrip('/')}{UPLOAD_ENDPOINT}"
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        data = {"model": MODEL_NAME}
        resp = requests.post(url, files=files, data=data, timeout=120)
    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("message") or err.get("error") or resp.text
        except ValueError:
            msg = resp.text
        raise RuntimeError(f"Upload failed ({resp.status_code}): {msg}")
    return resp.json()


def get_progress(api_base: str, task_id: str) -> dict:
    url = f"{api_base.rstrip('/')}{PROGRESS_ENDPOINT.format(task_id=task_id)}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("message") or err.get("error") or resp.text
        except ValueError:
            msg = resp.text
        raise RuntimeError(f"Progress request failed ({resp.status_code}): {msg}")
    return resp.json()


def poll_until_complete(api_base: str, task_id: str) -> dict:
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        payload = get_progress(api_base, task_id)
        error = payload.get("error")
        status = (
            payload.get("status") or payload.get("state") or payload.get("task_status") or "unknown"
        )
        complete = payload.get("complete") is True
        has_result = payload.get("result") is not None
        progress = payload.get("progress")
        if error:
            raise RuntimeError(f"Task failed: {error}")
        if status in ("failed", "error", "failure"):
            raise RuntimeError(f"Task failed: {payload.get('message') or 'unknown error'}")
        if status in ("completed", "success", "done", "finished", "complete"):
            return payload
        if complete and has_result:
            return payload
        if progress is not None and progress >= 100 and has_result:
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Task {task_id} did not complete within {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
    )


# ============================================================
# Result extraction
# ============================================================
def extract_burnout_level(payload: dict) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    def _dig(obj, *keys):
        cur = obj
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur
    candidates = [
        _dig(payload, "result", "result", "burnout_analysis"),
        _dig(payload, "result", "burnout_analysis"),
        _dig(payload, "burnout_analysis"),
        payload,
    ]
    for cand in candidates:
        if isinstance(cand, dict):
            level = cand.get("level")
            if level and isinstance(level, str):
                return level.lower()
            risk = cand.get("risk")
            if isinstance(risk, (int, float)):
                if risk < 0.35:
                    return "low"
                elif risk < 0.50:
                    return "moderate"
                elif risk < 0.65:
                    return "high"
                else:
                    return "severe"
    return "unknown"


# ============================================================
# Process audio files through API
# ============================================================
def process_dataset(api_base: str, dataset_name: str, directory: Path) -> list:
    """Returns [(filename, dataset, predicted_level), ...]."""
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"  [WARN] No supported audio files in {directory}")
        return []
    print(f"\n  Processing {dataset_name} ({len(files)} files from {directory})")
    results = []
    for i, fpath in enumerate(files, 1):
        label = f"{i}/{len(files)} {fpath.name}"
        print(f"    [{label}] uploading...", end=" ", flush=True)
        try:
            upload_resp = upload_audio(api_base, fpath)
        except (RuntimeError, requests.RequestException) as e:
            print(f"ERROR: {e}")
            results.append((fpath.name, dataset_name, "error"))
            continue
        task_id = upload_resp.get("task_id") or upload_resp.get("taskId")
        if not task_id:
            print("ERROR: no task_id")
            results.append((fpath.name, dataset_name, "error"))
            continue
        try:
            final = poll_until_complete(api_base, task_id)
        except (RuntimeError, TimeoutError) as e:
            print(f"ERROR: {e}")
            results.append((fpath.name, dataset_name, "error"))
            continue
        level = extract_burnout_level(final)
        print(level)
        results.append((fpath.name, dataset_name, level))
    return results


# ============================================================
# Labelled metrics computation
# ============================================================
def compute_labelled_metrics(predictions, ground_truth, threshold):
    """
    predictions: list of (filename, predicted_level)
    ground_truth: dict {filename_stem: 0|1}  (1 = burnout)
    threshold: minimum level to be considered positive
    """
    thresh_idx = LEVEL_ORDER.index(threshold)
    tp = fp = tn = fn = 0
    details = []

    for fname, pred_level in predictions:
        stem = Path(fname).stem
        y_true = ground_truth.get(stem, None)
        if y_true is None:
            continue
        pred_idx = LEVEL_ORDER.index(pred_level) if pred_level in LEVEL_ORDER else -1
        y_pred = 1 if pred_idx >= thresh_idx else 0

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

        details.append({
            "file": fname,
            "true": y_true,
            "predicted": pred_level,
            "pred_binary": y_pred,
        })

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total": total,
        "details": details,
    }


def print_labelled_report(metrics, threshold):
    print("=" * 60)
    print("  Burnout Model — Precision & Recall (labelled set)")
    print("=" * 60)
    print(f"\n  Threshold: ≥ '{threshold}' = POSITIVE")
    print(f"  Total files with labels: {metrics['total']}")
    print()
    print("  ┌──────────────────────┬───────┐")
    print("  │                      │ Count │")
    print("  ├──────────────────────┼───────┤")
    print(f"  │ True Positives (TP)  │ {metrics['tp']:>5} │")
    print(f"  │ False Positives (FP) │ {metrics['fp']:>5} │")
    print(f"  │ True Negatives (TN)  │ {metrics['tn']:>5} │")
    print(f"  │ False Negatives (FN) │ {metrics['fn']:>5} │")
    print("  └──────────────────────┴───────┘")
    print()
    print(f"  Precision  = {metrics['precision']:.4f}  ({metrics['tp']}/{metrics['tp'] + metrics['fp']})")
    print(f"  Recall     = {metrics['recall']:.4f}  ({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    print(f"  F1 Score   = {metrics['f1']:.4f}")
    print(f"  Accuracy   = {metrics['accuracy']:.4f}")
    print()

    # Per-class breakdown
    levels = [d["predicted"] for d in metrics["details"]]
    print("  ── Predicted distribution ──")
    for lvl in LEVEL_ORDER:
        cnt = levels.count(lvl)
        if cnt:
            print(f"    {lvl:<10} {cnt:>3}")
    print()


# ============================================================
# Load ground truth from TSV
# ============================================================
def load_ground_truth(tsv_path: Path) -> dict:
    """Returns dict {filename_stem: 1|0}."""
    gt = {}
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stem = row["Название файла звонка"].strip()
            label = 1 if row["Выгорел"].strip() == "Да" else 0
            gt[stem] = label
    return gt


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Measure precision/recall of the audio burnout model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help="Base URL of the API.")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL_DIR,
                        help="Directory with control recordings.")
    parser.add_argument("--burnout", type=Path, default=DEFAULT_BURNOUT_DIR,
                        help="Directory with burnout/test recordings.")
    parser.add_argument("--train", type=Path,
                        help="Directory with labelled training recordings.")
    parser.add_argument("--train-targets", type=Path,
                        help="TSV with ground truth (train_targets.tsv format).")
    parser.add_argument("--positive-threshold", default="high",
                        choices=LEVEL_ORDER,
                        help="Minimum level considered positive.")
    parser.add_argument("--output", type=Path,
                        help="CSV path to save per-file predictions.")
    args = parser.parse_args()

    # ── Labelled mode ──
    if args.train and args.train_targets:
        if not args.train.is_dir():
            print(f"[ERROR] Train directory not found: {args.train}")
            return 1
        if not args.train_targets.is_file():
            print(f"[ERROR] Targets file not found: {args.train_targets}")
            return 1

        # Load ground truth
        ground_truth = load_ground_truth(args.train_targets)
        print(f"Loaded {len(ground_truth)} ground truth labels ({sum(ground_truth.values())} positive, "
              f"{len(ground_truth) - sum(ground_truth.values())} negative)")

        # Process files
        predictions = process_dataset(args.api_base, "train", args.train)

        # Filter to only files with labels
        pred_with_gt = [(f, l) for f, ds, l in predictions if Path(f).stem in ground_truth]
        print(f"\n  Files with matching labels: {len(pred_with_gt)} / {len(predictions)}")

        # Compute metrics for multiple thresholds
        for thresh in ["low", "moderate", "high", "severe"]:
            metrics = compute_labelled_metrics(pred_with_gt, ground_truth, thresh)
            print_labelled_report(metrics, thresh)

        # Save CSV
        if args.output:
            with args.output.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "file", "true_label", "predicted_level", "pred_binary"
                ])
                w.writeheader()
                for d in metrics["details"]:
                    w.writerow({
                        "file": d["file"],
                        "true_label": d["true"],
                        "predicted_level": d["predicted"],
                        "pred_binary": d["pred_binary"],
                    })
            print(f"  Per-file results saved to: {args.output}")

    # ── Unlabelled mode (control / burnout) ──
    else:
        for name, path in [("control", args.control), ("burnout", args.burnout)]:
            if not path.is_dir():
                print(f"[ERROR] {name} directory not found: {path}")
                return 1

        all_results = []
        all_results.extend(process_dataset(args.api_base, "control", args.control))
        all_results.extend(process_dataset(args.api_base, "burnout", args.burnout))

        if not all_results:
            print("\n[ERROR] No results collected.")
            return 1

        thresh_idx = LEVEL_ORDER.index(args.positive_threshold)
        details = []
        for fname, ds, lvl in all_results:
            pred_idx = LEVEL_ORDER.index(lvl) if lvl in LEVEL_ORDER else -1
            flagged = 1 if pred_idx >= thresh_idx else 0
            details.append({"file": fname, "dataset": ds, "predicted": lvl, "flagged": flagged})

        print("=" * 60)
        print("  Burnout Model — Prediction Distribution Report")
        print("=" * 60)
        print("  NOTE: No ground truth labels. Distributions only.")
        print()

        for ds_name in ["control", "burnout"]:
            rows = [d for d in details if d["dataset"] == ds_name]
            if not rows:
                continue
            levels = [d["predicted"] for d in rows]
            flagged = sum(d["flagged"] for d in rows)
            total_ds = len(rows)
            label = "Control (assumed healthy)" if ds_name == "control" else "Burnout/test"
            print(f"  {label}  ({total_ds} files)")
            for lvl in LEVEL_ORDER:
                cnt = levels.count(lvl)
                if cnt:
                    print(f"    {lvl:<10} {cnt:>3}")
            print(f"    Flagged as ≥{args.positive_threshold}: {flagged}/{total_ds}  "
                  f"({100*flagged/total_ds:.1f}%)")
            print()

        if args.output:
            with args.output.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["file", "dataset", "predicted_level", "flagged"])
                w.writeheader()
                for d in details:
                    w.writerow(d)
            print(f"  Per-file results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
