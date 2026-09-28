#!/usr/bin/env python3
"""
Audio analysis client for razuma.tech — burnout and external-influence modes.

Usage:
    # Single file — burnout (default)
    python3 benchmark.py audio.mp3
    python3 benchmark.py audio.mp3 --tag                 # -> high

    # Single file — external influence (verdict: yes / no / maybe / unknown)
    python3 benchmark.py audio.mp3 --mode external_influence
    python3 benchmark.py audio.mp3 --mode external_influence --tag      # -> no

    # Batch
    python3 benchmark.py --batch train/no --jobs 4
    python3 benchmark.py --batch train/no --mode external_influence --jobs 4 --quiet --tags-only
"""

import argparse
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
PROGRESS_ENDPOINT = "/progress/{task_id}"
MODEL_NAME = "emotieff"

MODES = {
    "burnout": {
        "upload_endpoint": "/upload_burnout",
        "tag_field": "level",
        "title": "Burnout",
    },
    "external_influence": {
        "upload_endpoint": "/upload_external_influence",
        "tag_field": "status",
        "title": "External Influence",
    },
}

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 600
HEARTBEAT_EVERY_N_POLLS = 5

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".mp4", ".avi", ".webm"}
MAX_FILE_SIZE_MB = 50


# ============================================================
# External Influence — status → binary verdict
# ============================================================
# The server returns one of five statuses. We collapse them into a
# human-friendly verdict: yes / no / maybe / unknown.
#
#   LOW                         -> no       (score < 0.30)
#   ELEVATED_TENSION            -> no       (0.30 <= score < 0.50) — tension, not external
#   POSSIBLE_EXTERNAL_PRESSURE  -> maybe    (0.50 <= score < 0.80)
#   PROBABLE_EXTERNAL_INFLUENCE -> yes      (score >= 0.80)
#   INSUFFICIENT_DATA           -> unknown
INFLUENCE_VERDICT = {
    "LOW": "no",
    "ELEVATED_TENSION": "no",
    "POSSIBLE_EXTERNAL_PRESSURE": "maybe",
    "PROBABLE_EXTERNAL_INFLUENCE": "yes",
    "INSUFFICIENT_DATA": "unknown",
}


def external_influence_verdict(res: dict) -> str:
    """
    Derive a binary-ish verdict from the external-influence result.
    Returns one of: 'yes', 'no', 'maybe', 'unknown'.
    """
    if not isinstance(res, dict):
        return "unknown"

    status = (res.get("status") or "").strip().upper()
    if status in INFLUENCE_VERDICT:
        return INFLUENCE_VERDICT[status]

    # Fallback: use score if status is missing / unrecognized
    score = res.get("score")
    if isinstance(score, (int, float)):
        if score >= 0.80:
            return "yes"
        if score >= 0.50:
            return "maybe"
        return "no"

    return "unknown"


# ============================================================
# Helpers
# ============================================================
def print_banner(text: str, char: str = "=") -> None:
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def print_section(title: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def validate_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File is too large ({size_mb:.2f} MB). "
            f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
        )


# ============================================================
# API Calls
# ============================================================
def upload_audio(api_base: str, endpoint: str, file_path: Path, label: str = "") -> dict:
    url = f"{api_base.rstrip('/')}{endpoint}"
    prefix = f"[{label}] " if label else ""

    print_section(f"{prefix}Uploading File")
    print(f"  {prefix}Endpoint : {url}")
    print(f"  {prefix}File     : {file_path.name}")
    print(f"  {prefix}Size     : {format_bytes(file_path.stat().st_size)}")
    print(f"  {prefix}Model    : {MODEL_NAME}")

    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        data = {"model": MODEL_NAME}

        try:
            response = requests.post(url, files=files, data=data, timeout=120)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to {url}. Is the server running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Upload timed out.") from exc

    if not response.ok:
        try:
            err = response.json()
            msg = err.get("message") or err.get("error") or response.text
        except ValueError:
            msg = response.text
        raise RuntimeError(f"Upload failed ({response.status_code}): {msg}")

    return response.json()


def get_progress(api_base: str, task_id: str) -> dict:
    url = f"{api_base.rstrip('/')}{PROGRESS_ENDPOINT.format(task_id=task_id)}"
    response = requests.get(url, timeout=30)

    if not response.ok:
        try:
            err = response.json()
            msg = err.get("message") or err.get("error") or response.text
        except ValueError:
            msg = response.text
        raise RuntimeError(f"Progress request failed ({response.status_code}): {msg}")

    return response.json()


def poll_until_complete(
    api_base: str,
    task_id: str,
    label: str = "",
    max_attempts: int = MAX_POLL_ATTEMPTS,
) -> dict:
    prefix = f"[{label}] " if label else ""
    print_section(f"{prefix}Waiting for Analysis")
    print(f"  {prefix}Task ID : {task_id}")

    last_status = None
    last_progress = None

    for attempt in range(1, max_attempts + 1):
        try:
            payload = get_progress(api_base, task_id)
        except RuntimeError as exc:
            print(f"  {prefix}[{attempt:03d}] Warning: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        status = (
            payload.get("status")
            or payload.get("state")
            or payload.get("task_status")
            or "unknown"
        )
        progress = payload.get("progress")
        complete = payload.get("complete") is True
        error = payload.get("error")
        has_result = payload.get("result") is not None

        changed = (status != last_status) or (progress != last_progress)
        if changed:
            pct = f"{progress}%" if progress is not None else "?"
            print(f"  {prefix}[{attempt:03d}] status={status:<12} progress={pct}")
            last_status, last_progress = status, progress
        elif attempt % HEARTBEAT_EVERY_N_POLLS == 0:
            pct = f"{progress}%" if progress is not None else "?"
            waited = attempt * POLL_INTERVAL_SECONDS
            print(f"  {prefix}[{attempt:03d}] heartbeat ({waited}s) "
                  f"status={status} progress={pct}")

        if error:
            raise RuntimeError(f"Task failed: {error}")
        if status in ("failed", "error", "failure"):
            raise RuntimeError(
                f"Task failed: {payload.get('message') or 'unknown error'}"
            )

        if status in ("completed", "success", "done", "finished", "complete"):
            return payload
        if complete and has_result:
            return payload
        if progress is not None and progress >= 100 and has_result:
            print(f"  {prefix}[i] Detected completion via progress=100 with result.")
            return payload

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Task {task_id} did not complete within "
        f"{max_attempts * POLL_INTERVAL_SECONDS}s."
    )


# ============================================================
# Result Extraction — Burnout
# ============================================================
def extract_burnout_data(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}

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
        _dig(payload, "result", "result", "burnout"),
        _dig(payload, "result", "burnout"),
        _dig(payload, "burnout"),
        _dig(payload, "result", "result", "analysis"),
        _dig(payload, "result", "analysis"),
        _dig(payload, "analysis"),
    ]

    for cand in candidates:
        if isinstance(cand, dict) and (
            "state" in cand or "level" in cand or "components" in cand
        ):
            return cand

    if any(k in payload for k in ("state", "level", "components")):
        return payload

    return {}


def print_burnout_report(data: dict) -> None:
    if not data:
        print_section("Burnout Result")
        print("  No burnout data found in response.")
        return

    state = data.get("state", "UNKNOWN")
    level = data.get("level", "unknown")
    components = data.get("components", {}) or {}
    recommendations = data.get("recommendations", []) or []
    score = data.get("score")
    risk = data.get("risk")
    confidence = data.get("confidence")
    top_factor = data.get("top_factor")
    comment = data.get("comment")

    print_section("Burnout Analysis Result")
    print(f"  State       : {state}")
    print(f"  Level       : {level}")
    if score is not None:
        print(f"  Score       : {score}")
    if risk is not None:
        print(f"  Risk        : {risk}")
    if confidence is not None:
        print(f"  Confidence  : {confidence}")
    if top_factor:
        print(f"  Top Factor  : {top_factor}")
    if comment:
        print(f"  Comment     : {comment}")

    if components:
        print_section("Components")
        sorted_items = sorted(components.items(), key=lambda kv: kv[1] or 0, reverse=True)
        max_key_len = max(len(k) for k, _ in sorted_items)
        for key, value in sorted_items:
            try:
                p = float(value) * 100 if value <= 1 else float(value)
            except (TypeError, ValueError):
                p = 0.0
            bar_len = int(p / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            label = key.replace("_", " ").title()
            print(f"  {label:<{max_key_len + 5}} {p:6.2f}%  {bar}")

    if recommendations:
        print_section("Recommendations")
        for i, rec in enumerate(recommendations, 1):
            if isinstance(rec, dict):
                rec = rec.get("text") or rec.get("message") or json.dumps(rec)
            print(f"  {i}. {rec}")


# ============================================================
# Result Extraction — External Influence
# ============================================================
def extract_external_influence_data(payload: dict) -> dict:
    """
    External-influence response shape:

        payload.result            -> envelope { duration, sample_rate, result, diagnostics }
        payload.result.result     -> res { status, score, topFactors,
                                            persistentPattern, contextConfirmed,
                                            managerAction, confidence }
        payload.result.diagnostics-> diag { components, fragmentScores,
                                            rollingScores, audioQuality,
                                            contextFlags, insufficientReason,
                                            bestWindowIndex, windowSize, ... }
    """
    if not isinstance(payload, dict):
        return {}

    envelope = payload.get("result")
    if not isinstance(envelope, dict):
        return {}

    res = envelope.get("result") if isinstance(envelope.get("result"), dict) else None
    diag = envelope.get("diagnostics") if isinstance(envelope.get("diagnostics"), dict) else None

    if not res:
        return {}

    return {
        "envelope": envelope,
        "result": res,
        "diagnostics": diag or {},
    }


def print_external_influence_report(data: dict) -> None:
    if not data or not data.get("result"):
        print_section("External Influence Result")
        print("  No external-influence data found in response.")
        return

    envelope = data.get("envelope", {}) or {}
    res = data.get("result", {}) or {}
    diag = data.get("diagnostics", {}) or {}

    status = res.get("status", "UNKNOWN")
    verdict = external_influence_verdict(res)
    score = res.get("score")
    top_factors = res.get("topFactors", []) or []
    persistent = res.get("persistentPattern")
    context_confirmed = res.get("contextConfirmed")
    manager_action = res.get("managerAction")
    confidence = res.get("confidence")

    components = diag.get("components", {}) or {}
    context_flags = diag.get("contextFlags", []) or []
    audio_quality = diag.get("audioQuality")
    insufficient_reason = diag.get("insufficientReason")

    duration = envelope.get("duration")
    sample_rate = envelope.get("sample_rate")

    print_section("External Influence Result")
    print(f"  Verdict           : {verdict.upper()}")
    print(f"  Status            : {status}")
    if score is not None:
        print(f"  Score             : {score}")
    if confidence is not None:
        print(f"  Confidence        : {confidence}")
    if audio_quality is not None:
        print(f"  Audio Quality     : {audio_quality}")
    if duration is not None:
        print(f"  Duration          : {duration}")
    if sample_rate is not None:
        print(f"  Sample Rate       : {sample_rate}")
    if persistent is not None:
        print(f"  Persistent Pattern: {persistent}")
    if context_confirmed is not None:
        print(f"  Context Confirmed : {context_confirmed}")
    if manager_action:
        print(f"  Manager Action    : {manager_action}")
    if insufficient_reason:
        print(f"  Insufficient rsn  : {insufficient_reason}")

    if context_flags:
        print_section("Context Flags")
        for f in context_flags:
            print(f"  - {f}")

    if top_factors:
        print_section("Top Factors")
        for i, f in enumerate(top_factors, 1):
            print(f"  {i}. {f}")

    if components:
        print_section("Components")
        sorted_items = sorted(components.items(), key=lambda kv: kv[1] or 0, reverse=True)
        max_key_len = max(len(k) for k, _ in sorted_items)
        for key, value in sorted_items:
            try:
                p = float(value) * 100 if value <= 1 else float(value)
            except (TypeError, ValueError):
                p = 0.0
            bar_len = int(p / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            label = key.replace("_", " ").title()
            print(f"  {label:<{max_key_len + 5}} {p:6.2f}%  {bar}")


# ============================================================
# Emotion (burnout mode only, optional)
# ============================================================
def extract_emotion_data(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}

    inner = payload.get("result", {}).get("result", {}) if isinstance(payload.get("result"), dict) else {}
    main = inner.get("main_prediction") if isinstance(inner, dict) else None
    additional = inner.get("additional_probs") if isinstance(inner, dict) else None

    if not isinstance(main, dict):
        return {}

    return {
        "main": main,
        "additional": additional if isinstance(additional, dict) else {},
    }


def print_emotion_report(emotion: dict) -> None:
    if not emotion or not emotion.get("main"):
        return

    main = emotion["main"]
    label = main.get("label", "?")
    prob = main.get("probability")

    print_section("Emotion Summary")
    if prob is not None:
        print(f"  Main emotion : {label}  ({prob * 100:.2f}%)")
    else:
        print(f"  Main emotion : {label}")

    additional = emotion.get("additional") or {}
    if additional:
        print("  All emotions :")

        def _as_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        for key, value in sorted(additional.items(), key=lambda kv: _as_float(kv[1]), reverse=True):
            print(f"    - {key:<12} {_as_float(value) * 100:6.2f}%")


def print_full_json(payload: dict) -> None:
    print_section("Full JSON Response")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ============================================================
# Mode dispatch
# ============================================================
def extract_report(payload: dict, mode: str):
    if mode == "burnout":
        return print_burnout_report, extract_burnout_data(payload)
    if mode == "external_influence":
        return print_external_influence_report, extract_external_influence_data(payload)
    raise ValueError(f"Unknown mode: {mode}")


def extract_tag(payload: dict, mode: str) -> str:
    """
    Return the single-word tag for --tag and batch output.
      burnout            -> level  (low / moderate / high / severe / unknown)
      external_influence -> verdict (yes / no / maybe / unknown)
    """
    if mode == "burnout":
        data = extract_burnout_data(payload)
        return (data.get("level") or "unknown").strip()

    if mode == "external_influence":
        data = extract_external_influence_data(payload)
        res = data.get("result") or {}
        return external_influence_verdict(res)

    return "unknown"


# ============================================================
# Core pipeline (single file)
# ============================================================
def process_one(
    api_base: str,
    file_path: Path,
    mode: str,
    label: str = "",
    max_attempts: int = MAX_POLL_ATTEMPTS,
) -> str:
    try:
        validate_file(file_path)
    except (FileNotFoundError, ValueError) as exc:
        return f"ERROR:{exc}"

    endpoint = MODES[mode]["upload_endpoint"]

    try:
        upload_response = upload_audio(api_base, endpoint, file_path, label=label)
    except RuntimeError as exc:
        return f"ERROR:{exc}"

    task_id = upload_response.get("task_id") or upload_response.get("taskId")
    if not task_id:
        return "ERROR:no task_id"

    try:
        final_payload = poll_until_complete(
            api_base, task_id, label=label, max_attempts=max_attempts
        )
    except (RuntimeError, TimeoutError) as exc:
        return f"ERROR:{exc}"

    return extract_tag(final_payload, mode)


# ============================================================
# Batch Driver
# ============================================================
def run_batch(
    api_base: str,
    folder: Path,
    mode: str,
    jobs: int = 1,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    tags_only: bool = False,
    quiet: bool = False,
) -> int:
    # ── MUST be first statement in this function ──────────
    global print
    _real_print = print

    def print(*a, **kw):  # noqa: A001
        if quiet:
            return
        kw.setdefault("file", sys.stderr)
        _real_print(*a, **kw)
    # ───────────────────────────────────────────────────────

    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        return 1

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"[ERROR] No supported audio files in {folder}")
        return 1

    print(f"[i] Batch: {len(files)} file(s) in {folder} (mode={mode}, jobs={jobs})")

    def _emit(fname: str, tag: str) -> None:
        if tags_only:
            sys.stdout.write(f"{tag}\n")
        else:
            sys.stdout.write(f"{fname}\t{tag}\n")
        sys.stdout.flush()

    if jobs <= 1:
        for i, f in enumerate(files, 1):
            label = "" if quiet else f"{i}/{len(files)} {f.name}"
            tag = process_one(api_base, f, mode, label=label, max_attempts=max_attempts)
            _emit(f.name, tag)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        labeled = [
            (("" if quiet else f"{i}/{len(files)} {f.name}"), f)
            for i, f in enumerate(files, 1)
        ]
        results = {}

        def _work(label: str, path: Path) -> tuple:
            return path, process_one(api_base, path, mode, label=label,
                                     max_attempts=max_attempts)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_work, lbl, f): f for lbl, f in labeled}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    path, tag = fut.result()
                    results[path] = tag
                except Exception as exc:
                    results[f] = f"ERROR:{exc}"

        for f in files:
            _emit(f.name, results.get(f, "ERROR"))

    return 0


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an audio file for analysis and print the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file", type=Path, nargs="?",
        help="Path to the audio file. Optional when using --batch.",
    )
    parser.add_argument(
        "--mode", choices=sorted(MODES.keys()), default="burnout",
        help="Analysis mode.",
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="Base URL of the API.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full raw JSON response at the end.",
    )
    parser.add_argument(
        "--emotions", action="store_true",
        help="Also print the emotion summary (burnout mode only).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional path to save the full JSON result to a file.",
    )
    parser.add_argument(
        "--tag", action="store_true",
        help="Print only the final tag (level / verdict) to stdout and exit.",
    )
    parser.add_argument(
        "--batch", type=Path, metavar="DIR",
        help="Process every supported audio file in DIR.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of files to process in parallel in --batch mode.",
    )
    parser.add_argument(
        "--max-wait", type=float, default=MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS,
        help="Maximum seconds to wait per file before giving up.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress ALL non-result output (banners, progress, heartbeats).",
    )
    parser.add_argument(
        "--tags-only", action="store_true",
        help="In --batch mode, print only the tag per file (no filename).",
    )
    return parser.parse_args()


# ============================================================
# Entrypoint
# ============================================================
def main() -> int:
    args = parse_args()
    max_attempts = max(1, int(args.max_wait / POLL_INTERVAL_SECONDS))

    # ── Batch mode ────────────────────────────────────────
    if args.batch is not None:
        return run_batch(
            args.api_base,
            args.batch,
            args.mode,
            args.jobs,
            max_attempts,
            tags_only=args.tags_only,
            quiet=args.quiet,
        )

    # ── Single-file mode ──────────────────────────────────
    if args.tag:
        global print
        _real_print = print

        def print(*a, **kw):  # noqa: A001
            kw.setdefault("file", sys.stderr)
            _real_print(*a, **kw)

    if args.file is None:
        print("[ERROR] No file given. Pass a file or use --batch DIR.",
              file=sys.stderr)
        return 1

    print_banner(f"{MODES[args.mode]['title']} Audio Analysis")

    try:
        validate_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    endpoint = MODES[args.mode]["upload_endpoint"]

    # ---- Upload ----
    try:
        upload_response = upload_audio(args.api_base, endpoint, args.file)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    task_id = upload_response.get("task_id") or upload_response.get("taskId")
    if not task_id:
        print("\n[ERROR] Server response did not include a task_id.", file=sys.stderr)
        print(json.dumps(upload_response, indent=2), file=sys.stderr)
        return 1

    # ---- Poll ----
    try:
        final_payload = poll_until_complete(
            args.api_base, task_id, max_attempts=max_attempts
        )
    except (RuntimeError, TimeoutError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    # ---- Tag mode: print ONLY the verdict/level to stdout ----
    if args.tag:
        sys.stdout.write(extract_tag(final_payload, args.mode) + "\n")
        return 0

    # ---- Full report ----
    print_fn, data = extract_report(final_payload, args.mode)
    print_fn(data)

    if args.emotions and args.mode == "burnout":
        print_emotion_report(extract_emotion_data(final_payload))

    if args.json:
        print_full_json(final_payload)

    if args.output:
        args.output.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[i] Full JSON saved to: {args.output}")

    print_banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())