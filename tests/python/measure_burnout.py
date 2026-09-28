#!/usr/bin/env python3
"""
Measure precision and recall of the audio burnout model via the server API.

Usage:
    # Default — uses VITE_API_URL env or https://razuma.tech/api
    python3 tests/python/measure_burnout.py

    # Custom API base
    python3 tests/python/measure_burnout.py --api-base http://localhost:80/api

    # Custom data directories
    python3 tests/python/measure_burnout.py \
        --control /path/to/control \
        --burnout /path/to/burnout/test

    # Adjust positive threshold (default: ≥"high" is positive)
    python3 tests/python/measure_burnout.py --positive-threshold high

    # Save per-file results to CSV
    python3 tests/python/measure_burnout.py --output results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================
DEFAULT_API_BASE = os.environ.get("VITE_API_URL", "https://razuma.tech/api")
UPLOAD_ENDPOINT = "/upload_burnout"
PROGRESS_ENDPOINT = "/progress/{task_id}"
MODEL_NAME = "emotieff"

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 600          # 20 min per file

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".mp4", ".avi", ".webm"}
MAX_FILE_SIZE_MB = 50

# Default data paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONTROL_DIR = REPO_ROOT / "tests" / "data" / "control"
DEFAULT_BURNOUT_DIR = REPO_ROOT / "tests" / "data" / "burnout" / "test"

# Burnout levels in ascending order of severity
LEVEL_ORDER = ["low", "moderate", "high", "severe"]


# ============================================================
# Note on ground truth
#
# These datasets are NOT labelled. "control" is assumed healthy
# (for false-positive rate estimation), and "burnout/test" has
# UNKNOWN labels. Precision, recall, F1, and accuracy require
# known ground truth and CANNOT be computed from this data.
# Only prediction distributions and control false-positive rate
# are reported.
# ============================================================


# ============================================================
# API helpers
# ============================================================
def upload_audio(api_base: str, file_path: Path) -> dict:
    """Upload a WAV file for burnout analysis. Returns JSON with task_id."""
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
    """Poll the progress endpoint."""
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
    """Poll until the task completes or fails."""
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        payload = get_progress(api_base, task_id)

        error = payload.get("error")
        status = (
            payload.get("status")
            or payload.get("state")
            or payload.get("task_status")
            or "unknown"
        )
        complete = payload.get("complete") is True
        has_result = payload.get("result") is not None
        progress = payload.get("progress")

        # Terminal: failure
        if error:
            raise RuntimeError(f"Task failed: {error}")
        if status in ("failed", "error", "failure"):
            raise RuntimeError(f"Task failed: {payload.get('message') or 'unknown error'}")

        # Terminal: success
        if status in ("completed", "success", "done", "finished", "complete"):
            return payload
        if complete and has_result:
            return payload
        if progress is not None and progress >= 100 and has_result:
            return payload

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Task {task_id} did not complete within "
        f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
    )


# ============================================================
# Result extraction
# ============================================================
def extract_burnout_level(payload: dict) -> str:
    """Dig through the nested response to find the burnout level."""
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
            # Fallback: derive from risk score
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
# Metrics computation
# ============================================================
def compute_stats(results, flag_threshold: str):
    """
    Compute prediction distributions per dataset and control false-positive rate.
    
    results: list of (filename, dataset, predicted_level)
    
    NO precision/recall: required ground truth labels don't exist.
    """
    threshold_idx = LEVEL_ORDER.index(flag_threshold)
    details = []

    for fname, dataset, pred_level in results:
        pred_idx = LEVEL_ORDER.index(pred_level) if pred_level in LEVEL_ORDER else -1
        flagged = 1 if pred_idx >= threshold_idx else 0
        details.append({
            "file": fname,
            "dataset": dataset,
            "predicted": pred_level,
            "flagged": flagged,
        })

    return {"total": len(results), "flag_threshold": flag_threshold, "details": details}


def print_report(stats):
    """Print prediction distributions and control false-positive rate."""
    threshold = stats["flag_threshold"]
    print("=" * 60)
    print("  Burnout Model — Prediction Distribution Report")
    print("=" * 60)
    print("  NOTE: No ground truth labels exist. Precision/recall")
    print("        cannot be computed. Only distributions and")
    print("        control false-positive rate are reported.")
    print()

    for dataset_name in ["control", "burnout"]:
        rows = [d for d in stats["details"] if d["dataset"] == dataset_name]
        if not rows:
            continue
        levels = [d["predicted"] for d in rows]
        flagged = sum(d["flagged"] for d in rows)
        total_ds = len(rows)

        label = "Control (assumed healthy)" if dataset_name == "control" else "Burnout/test (UNKNOWN labels)"
        print(f"  {label}  ({total_ds} files)")
        for lvl in LEVEL_ORDER:
            cnt = levels.count(lvl)
            if cnt:
                bar = "█" * cnt
                print(f"    {lvl:<10} {cnt:>3}  {bar}")
        print(f"    Flagged as ≥{threshold}: {flagged}/{total_ds}  ({100*flagged/total_ds:.1f}%)")
        print()

    # Control false-positive rate
    control_flagged = sum(d["flagged"] for d in stats["details"] if d["dataset"] == "control")
    control_total = sum(1 for d in stats["details"] if d["dataset"] == "control")
    print(f"  Control false-positive rate (≥{threshold}): "
          f"{control_flagged}/{control_total} = {100*control_flagged/control_total:.1f}%")
    print()


# ============================================================
# Main pipeline
# ============================================================
def process_dataset(
    api_base: str,
    dataset_name: str,
    directory: Path,
) -> list:
    """Process all audio files in a directory. Returns [(filename, dataset, level), ...]."""
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
            print("ERROR: no task_id in response")
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


def main():
    parser = argparse.ArgumentParser(
        description="Measure precision/recall of the audio burnout model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="Base URL of the API (e.g. https://razuma.tech/api).",
    )
    parser.add_argument(
        "--control", type=Path, default=DEFAULT_CONTROL_DIR,
        help=f"Directory with control (no burnout) recordings.",
    )
    parser.add_argument(
        "--burnout", type=Path, default=DEFAULT_BURNOUT_DIR,
        help=f"Directory with burnout-positive recordings.",
    )
    parser.add_argument(
        "--positive-threshold", default="high",
        choices=LEVEL_ORDER,
        help="Minimum level considered positive (e.g. 'high' = high & severe → positive).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional CSV path to save per-file predictions.",
    )
    args = parser.parse_args()

    # Validate directories
    for name, path in [("control", args.control), ("burnout", args.burnout)]:
        if not path.is_dir():
            print(f"[ERROR] {name} directory not found: {path}")
            return 1

    # Process both datasets
    all_results = []
    all_results.extend(process_dataset(args.api_base, "control", args.control))
    all_results.extend(process_dataset(args.api_base, "burnout", args.burnout))

    if not all_results:
        print("\n[ERROR] No results collected.")
        return 1

    # Compute and print stats
    stats = compute_stats(all_results, args.positive_threshold)
    print_report(stats)

    # Save per-file CSV if requested
    if args.output:
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "dataset", "predicted_level", "flagged"
            ])
            writer.writeheader()
            for d in stats["details"]:
                writer.writerow({
                    "file": d["file"],
                    "dataset": d["dataset"],
                    "predicted_level": d["predicted"],
                    "flagged": d["flagged"],
                })
        print(f"\n  Per-file results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
