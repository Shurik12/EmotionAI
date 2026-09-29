#!/usr/bin/env python3
"""
Process video files through the EmotionAI API and save average emotion
probabilities (including the dominant emotion) for each video.

Usage:
    # Process first 10 videos
    python3 tests/python/process_videos.py

    # Process all videos
    python3 tests/python/process_videos.py --all

    # Custom API base
    python3 tests/python/process_videos.py --api-base http://localhost:80/api

    # Save CSV to a custom path
    python3 tests/python/process_videos.py --output my_results.csv
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
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 600

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VIDEO_DIR = REPO_ROOT / "tests" / "data" / "videos"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "video_results.csv"

# Standard 8 emotion labels from Image.cpp
EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "happiness",
                  "neutral", "sadness", "surprise"]


# ============================================================
# API helpers
# ============================================================
def upload_video(api_base: str, file_path: Path) -> dict:
    """Upload a video to the API and return the JSON response with task_id."""
    url = f"{api_base.rstrip('/')}/upload"
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"    Uploading ({file_size_mb:.1f} MiB)...", end=" ", flush=True)

    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        data = {"model": "emotieff"}
        resp = requests.post(url, files=files, data=data, timeout=300)

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("message") or err.get("error") or resp.text
        except ValueError:
            msg = resp.text
        raise RuntimeError(f"Upload failed ({resp.status_code}): {msg}")

    return resp.json()


def get_progress(api_base: str, task_id: str) -> dict:
    """Poll progress for a task ID."""
    url = f"{api_base.rstrip('/')}/progress/{task_id}"
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
    """Poll task progress until complete or failed."""
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        payload = get_progress(api_base, task_id)
        error = payload.get("error")
        status = (
            payload.get("status") or payload.get("state")
            or payload.get("task_status") or "unknown"
        )
        complete = payload.get("complete") is True
        progress = payload.get("progress")

        if error and error is not None:
            raise RuntimeError(f"Task failed: {error}")
        if status in ("failed", "error", "failure"):
            msg = payload.get("message") or "unknown error"
            raise RuntimeError(f"Task failed: {msg}")
        if complete:
            return payload
        if progress is not None and progress >= 100:
            return payload

        if attempt % 10 == 0:
            print(f"\r    Progress: {progress}% (attempt {attempt})...", end=" ", flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Task {task_id} did not complete within "
        f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
    )


# ============================================================
# Result extraction helpers
# ============================================================
def extract_average_emotions(payload: dict) -> dict:
    """
    Extract average emotions from the completed task payload.

    The payload structure for video processing (from FileProcessor.cpp):
    {
        "type": "video",
        "frames_processed": N,
        "results": [...],
        "average_emotions": {
            "anger": 0.05,
            "contempt": 0.01,
            ...
        },
        "average_main_emotion": {
            "label": "neutral",
            "probability": 0.45
        },
        ...
    }
    """
    # The average_emotions may be at the top level, or nested inside
    for candidate_key in ("average_emotions", "avg_emotions"):
        avg = payload.get(candidate_key)
        if avg and isinstance(avg, dict):
            return _normalize_emotions(avg)

    # Try digging into result sub-objects
    for key in ("result", "data"):
        sub = payload.get(key)
        if sub and isinstance(sub, dict):
            for candidate_key in ("average_emotions", "avg_emotions"):
                avg = sub.get(candidate_key)
                if avg and isinstance(avg, dict):
                    return _normalize_emotions(avg)

    return {}


def extract_average_main_emotion(payload: dict) -> dict:
    """Extract the average main emotion (label + probability)."""
    for candidate_key in ("average_main_emotion", "avg_main_emotion"):
        me = payload.get(candidate_key)
        if me and isinstance(me, dict):
            return {
                "label": me.get("label", "unknown"),
                "probability": me.get("probability", 0.0),
            }

    # Fall back to derived from average_emotions
    avg_emotions = extract_average_emotions(payload)
    if avg_emotions:
        best_label = max(avg_emotions, key=avg_emotions.get)
        return {
            "label": best_label,
            "probability": avg_emotions[best_label],
        }

    return {"label": "unknown", "probability": 0.0}


def _normalize_emotions(emotions: dict) -> dict:
    """Convert all values to float (rounded to 2 decimal places) and ensure standard label keys exist."""
    result = {}
    for label in EMOTION_LABELS:
        val = emotions.get(label)
        if val is not None:
            try:
                result[label] = round(float(val), 2)
            except (ValueError, TypeError):
                result[label] = 0.0
        else:
            result[label] = 0.0
    return result


# ============================================================
# Video processing
# ============================================================
def process_videos(
    api_base: str,
    video_dir: Path,
    limit: int | None = None,
) -> list[dict]:
    """
    Process videos from the given directory.

    Returns a list of dicts with keys:
      filename, type, frames_processed, total_frames, fps, duration,
      main_emotion, main_probability,
      avg_anger, avg_contempt, avg_disgust, avg_fear,
      avg_happiness, avg_neutral, avg_sadness, avg_surprise
    """
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".webm", ".mov", ".mkv")
    )

    if not videos:
        print("  [WARN] No video files found.")
        return []

    if limit is not None:
        videos = videos[:limit]

    print(f"\n  Processing {len(videos)} video(s) from {video_dir}")
    results = []

    for i, fpath in enumerate(videos, 1):
        label = f"{i}/{len(videos)} {fpath.name}"
        print(f"  [{label}]")

        row = None  # result row; set on success

        for attempt in range(1, 3):  # retry once on error
            if attempt > 1:
                print(f"    Retry ({attempt}/2)...", end=" ", flush=True)

            try:
                upload_resp = upload_video(api_base, fpath)
            except (RuntimeError, requests.RequestException) as e:
                print(f"ERROR: {e}")
                if attempt < 2:
                    continue
                _append_error_result(results, fpath.name, str(e))
                break

            task_id = upload_resp.get("task_id") or upload_resp.get("taskId")
            if not task_id:
                print("ERROR: no task_id in response")
                if attempt < 2:
                    continue
                _append_error_result(results, fpath.name, "no task_id")
                break

            print(f"task_id={task_id[:8]}...", end=" ", flush=True)

            try:
                final = poll_until_complete(api_base, task_id)
            except (RuntimeError, TimeoutError) as e:
                print(f"ERROR: {e}")
                if attempt < 2:
                    continue
                _append_error_result(results, fpath.name, str(e))
                break

            # Extract results
            file_type = final.get("type", "unknown")
            frames_processed = final.get("frames_processed", 0)
            total_frames = final.get("total_frames", 0)
            fps = final.get("fps", 0)
            duration = final.get("duration", 0)

            avg_emotions = extract_average_emotions(final)
            main_emotion = extract_average_main_emotion(final)

            print(f"done — {main_emotion['label']} ({main_emotion['probability']:.2f})")

            row = {
                "filename": fpath.name,
                "type": file_type,
                "frames_processed": frames_processed,
                "total_frames": total_frames,
                "fps": round(fps, 2) if isinstance(fps, float) else fps,
                "duration": round(duration, 2) if isinstance(duration, float) else duration,
                "main_emotion": main_emotion["label"],
                "main_probability": round(main_emotion["probability"], 2),
            }
            for emo_label in EMOTION_LABELS:
                row[f"avg_{emo_label}"] = round(avg_emotions.get(emo_label, 0.0), 2)

            break  # success — exit retry loop

        if row is not None:
            results.append(row)

    return results


def _append_error_result(results: list, filename: str, error: str):
    """Append an error row to results."""
    row = {
        "filename": filename,
        "type": "error",
        "frames_processed": 0,
        "total_frames": 0,
        "fps": 0,
        "duration": 0,
        "main_emotion": "error",
        "main_probability": 0.0,
    }
    for label in EMOTION_LABELS:
        row[f"avg_{label}"] = 0.0
    row["error"] = error
    results.append(row)


# ============================================================
# CSV output
# ============================================================
def save_results(results: list[dict], output_path: Path):
    """Save results to a CSV file."""
    fieldnames = [
        "filename", "type", "frames_processed", "total_frames",
        "fps", "duration", "main_emotion", "main_probability",
    ] + [f"avg_{label}" for label in EMOTION_LABELS] + ["error"]

    # Filter to fieldnames that actually appear
    present_fields = [f for f in fieldnames if any(f in r for r in results)]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=present_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Results saved to: {output_path}")


def print_summary(results: list[dict]):
    """Print a brief summary table."""
    ok = [r for r in results if r.get("type") != "error"]
    errs = [r for r in results if r.get("type") == "error"]

    print()
    print("=" * 70)
    print("  Video Emotion Analysis — Summary")
    print("=" * 70)
    print(f"\n  Total: {len(results)}  |  OK: {len(ok)}  |  Errors: {len(errs)}")
    print()

    if ok:
        header = f"  {'File':<40} {'Main Emotion':<20} {'Main Prob':>10}"
        print(header)
        print("  " + "-" * 70)
        for r in ok:
            fname = r["filename"]
            if len(fname) > 38:
                fname = fname[:35] + "..."
            print(f"  {fname:<40} {r['main_emotion']:<20} {r['main_probability']:>10.4f}")

    print()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Process videos and extract average emotion probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="Base URL of the API (e.g. https://razuma.tech/api or http://localhost:80/api).",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path to save the CSV results.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process ALL videos (default: first 10 only).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Explicit limit on number of videos to process (overrides --all).",
    )
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        print(f"[ERROR] Video directory not found: {args.video_dir}")
        return 1

    # Determine limit
    if args.limit is not None:
        limit = args.limit
    elif args.all:
        limit = None  # no limit
    else:
        limit = 10  # default: first 10

    print(f"API base: {args.api_base}")
    print(f"Video directory: {args.video_dir}")
    print(f"Limit: {'all' if limit is None else limit}")
    print()

    results = process_videos(
        api_base=args.api_base,
        video_dir=args.video_dir,
        limit=limit,
    )

    if not results:
        print("\n[ERROR] No results collected.")
        return 1

    save_results(results, args.output)
    print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
