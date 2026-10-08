#!/usr/bin/env python3
"""
Measure video processing speed through the EmotionAI API.

Usage:
    python3 tests/python/measure_video_speed.py

Environment:
    VITE_API_URL - API base URL (default: http://localhost:8080/api)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_API_BASE = os.environ.get("VITE_API_URL", "http://localhost:8080/api")
POLL_INTERVAL_SECONDS = 1
MAX_POLL_ATTEMPTS = 3600  # up to 1 hour per video

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_DIR = REPO_ROOT / "videos" / "Видео_Исследование"


def upload_video(api_base: str, file_path: Path) -> dict:
    """Upload a video to the API and return the JSON response with task_id."""
    url = f"{api_base.rstrip('/')}/upload"
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"    Uploading ({file_size_mb:.1f} MiB)...", end=" ", flush=True)

    t_start = time.monotonic()
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        data = {"model": "emotieff"}
        resp = requests.post(url, files=files, data=data, timeout=600)
    t_upload = time.monotonic() - t_start

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("message") or err.get("error") or resp.text
        except ValueError:
            msg = resp.text
        raise RuntimeError(f"Upload failed ({resp.status_code}): {msg}")

    result = resp.json()
    result["_upload_time_sec"] = round(t_upload, 3)
    return result


def poll_until_complete(api_base: str, task_id: str) -> dict:
    """Poll task progress until complete or failed."""
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        t_start = time.monotonic()
        url = f"{api_base.rstrip('/')}/progress/{task_id}"
        resp = requests.get(url, timeout=30)
        poll_time = time.monotonic() - t_start

        if not resp.ok:
            try:
                err = resp.json()
                msg = err.get("message") or err.get("error") or resp.text
            except ValueError:
                msg = resp.text
            raise RuntimeError(f"Progress request failed ({resp.status_code}): {msg}")

        payload = resp.json()
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
            payload["_poll_time_sec"] = round(poll_time, 3)
            return payload
        if progress is not None and progress >= 100:
            payload["_poll_time_sec"] = round(poll_time, 3)
            return payload

        if attempt % 10 == 0:
            elapsed = attempt * POLL_INTERVAL_SECONDS
            print(f"\r    Waiting... {progress}% (elapsed: {elapsed}s)...", end=" ", flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Task {task_id} did not complete within "
        f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
    )


EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "happiness",
                  "neutral", "sadness", "surprise"]


def extract_results(payload: dict) -> dict:
    """Extract key metrics from completed task payload."""
    return {
        "type": payload.get("type", "unknown"),
        "frames_processed": payload.get("frames_processed", 0),
        "total_frames": payload.get("total_frames", 0),
        "fps": payload.get("fps", 0),
        "duration": payload.get("duration", 0),
        "average_main_emotion": payload.get("average_main_emotion", {}),
    }


def measure_video(api_base: str, file_path: Path) -> dict:
    """Measure upload + processing time for one video."""
    label = file_path.name
    print(f"\n{'='*70}")
    print(f"  [{label}]")
    print(f"  Size: {file_path.stat().st_size / (1024*1024):.1f} MiB")

    upload_resp = upload_video(api_base, file_path)
    task_id = upload_resp.get("task_id") or upload_resp.get("taskId")
    upload_time = upload_resp.get("_upload_time_sec", 0)

    if not task_id:
        raise RuntimeError("No task_id in upload response")

    print(f"task_id={task_id[:8]}... (upload: {upload_time:.1f}s)", flush=True)

    t_proc_start = time.monotonic()
    final = poll_until_complete(api_base, task_id)
    t_processing = time.monotonic() - t_proc_start

    results = extract_results(final)
    total_time = upload_time + t_processing

    frames = results["frames_processed"]
    vid_duration = results["duration"]
    vid_fps = results["fps"]

    time_per_frame = t_processing / frames if frames > 0 else 0
    frames_per_sec = frames / t_processing if t_processing > 0 else 0
    speed_ratio = vid_duration / t_processing if t_processing > 0 else 0  # video duration / processing time

    main_emo = results.get("average_main_emotion", {})
    main_label = main_emo.get("label", "?")
    main_prob = main_emo.get("probability", 0)

    print(f"  {'Done!':<8} frames={frames} | main={main_label} ({main_prob:.2f})")
    print(f"  {'Timing:':<8} upload={upload_time:.1f}s | processing={t_processing:.1f}s | total={total_time:.1f}s")
    print(f"  {'Speed:':<8} {time_per_frame:.3f}s/frame | {frames_per_sec:.1f} frames/s | {speed_ratio:.1f}x realtime")

    return {
        "filename": label,
        "file_size_mb": round(file_path.stat().st_size / (1024*1024), 1),
        "video_duration_sec": round(vid_duration, 1) if isinstance(vid_duration, (int, float)) else 0,
        "video_fps": round(vid_fps, 2) if isinstance(vid_fps, (int, float)) else 0,
        "total_frames": results["total_frames"],
        "frames_processed": frames,
        "upload_time_sec": round(upload_time, 2),
        "processing_time_sec": round(t_processing, 2),
        "total_time_sec": round(total_time, 2),
        "time_per_frame_ms": round(time_per_frame * 1000, 2),
        "frames_per_sec": round(frames_per_sec, 2),
        "speed_vs_realtime": round(speed_ratio, 2),
        "main_emotion": main_label,
        "main_emotion_prob": round(main_prob, 4),
    }


def print_summary(results: list[dict]):
    """Print a formatted summary table."""
    print()
    print("=" * 100)
    print("  VIDEO PROCESSING SPEED BENCHMARK — SUMMARY")
    print("=" * 100)
    print()
    print(f"  {'File':<40} {'Duration':>9} {'Frames':>7} {'Proc':>7} {'Upload':>8} {'Process':>9} {'Total':>9}  {'ms/fr':>7} {'fr/s':>7} {'Speed':>6}")
    print(f"  {'-'*38} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*9}  {'-'*7} {'-'*7} {'-'*6}")
    for r in results:
        fname = r["filename"]
        if len(fname) > 38:
            fname = fname[:35] + "..."
        print(f"  {fname:<40} {r['video_duration_sec']:>9.1f}s {r['total_frames']:>7} {r['frames_processed']:>7} "
              f"{r['upload_time_sec']:>8.1f}s {r['processing_time_sec']:>9.1f}s {r['total_time_sec']:>9.1f}s  "
              f"{r['time_per_frame_ms']:>7.2f} {r['frames_per_sec']:>7.1f} {r['speed_vs_realtime']:>5.1f}x")

    # Averages
    avg_proc = sum(r["processing_time_sec"] for r in results) / len(results)
    avg_fr = sum(r["frames_per_sec"] for r in results) / len(results)
    avg_ms = sum(r["time_per_frame_ms"] for r in results) / len(results)
    avg_ratio = sum(r["speed_vs_realtime"] for r in results) / len(results)
    print(f"  {'─'*40} {'─'*9} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*9}  {'─'*7} {'─'*7} {'─'*6}")
    print(f"  {'AVERAGE':<40} {'':>9} {'':>7} {'':>7} {'':>8} {avg_proc:>9.1f}s {'':>9}  {avg_ms:>7.2f} {avg_fr:>7.1f} {avg_ratio:>5.1f}x")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Measure video processing speed through the EmotionAI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="Base URL of the API.",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=VIDEO_DIR,
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Save results as JSON to this path.",
    )
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        print(f"[ERROR] Video directory not found: {args.video_dir}")
        return 1

    # Find only MP4 files with video streams
    videos = sorted(
        p for p in args.video_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    )

    if not videos:
        print(f"[ERROR] No MP4 video files found in {args.video_dir}")
        return 1

    print(f"API base: {args.api_base}")
    print(f"Video directory: {args.video_dir}")
    print(f"Videos found: {len(videos)}")
    for v in videos:
        size_mb = v.stat().st_size / (1024*1024)
        print(f"  - {v.name} ({size_mb:.1f} MiB)")

    print(f"\n{'#'*70}")
    print(f"  Starting benchmark — processing {len(videos)} video(s)")
    print(f"{'#'*70}")

    results = []
    for i, fpath in enumerate(videos, 1):
        print(f"\n--- Video {i}/{len(videos)} ---")
        try:
            result = measure_video(args.api_base, fpath)
            results.append(result)
        except (RuntimeError, requests.RequestException, TimeoutError) as e:
            print(f"  ERROR: {e}")
            results.append({
                "filename": fpath.name,
                "file_size_mb": round(fpath.stat().st_size / (1024*1024), 1),
                "error": str(e),
            })

    print_summary(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
