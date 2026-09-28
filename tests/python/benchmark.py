#!/usr/bin/env python3
"""
Burnout Audio Analysis Client
Uploads an audio file to the server and prints burnout analysis results.

Usage:
    # Single file — full report
    python3 benchmark.py audio.mp3

    # Single file — full report + emotion summary
    python3 benchmark.py audio.mp3 --emotions

    # Single file — full report + raw JSON
    python3 benchmark.py audio.mp3 --json

    # Single file — print ONLY the burnout level to stdout
    python3 benchmark.py audio.mp3 --tag
    # -> "high"

    # Single file — write filename/verdict/description to a TSV result file
    python3 benchmark.py audio.mp3 --result-file result.tsv

    # Batch — default: "<filename>\\t<tag>" on stdout, chatter on stderr
    python3 benchmark.py --batch train/no
    python3 benchmark.py --batch train/no --jobs 4

    # Batch — also write filename/verdict/description to a TSV result file
    python3 benchmark.py --batch train/no --jobs 4 --result-file results.tsv

    # Batch — only the tag per file, no chatter at all
    python3 benchmark.py --batch train/no --jobs 4 --quiet --tags-only
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from threading import Lock

import requests


# ============================================================
# Configuration
# ============================================================
DEFAULT_API_BASE = os.environ.get("VITE_API_URL", "https://razuma.tech/api")
UPLOAD_ENDPOINT = "/upload_burnout"
PROGRESS_ENDPOINT = "/progress/{task_id}"
MODEL_NAME = "emotieff"

POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 600          # 20 min default per file
HEARTBEAT_EVERY_N_POLLS = 5      # heartbeat every N polls (when not quiet)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".mp4", ".avi", ".webm"}
MAX_FILE_SIZE_MB = 50


# ============================================================
# Burnout state texts (RU) — from translations.js → burnout.*
# ============================================================

# burnout.states.*
STATE_NAMES = {
    "normal": "Стабильное состояние",
    "shortStress": "Кратковременное напряжение",
    "sustainedStress": "Повышенное напряжение",
    "burnoutLike": "Состояние, похожее на выгорание",
    "lowAffect": "Сниженная эмоциональная выразительность",
    "insufficientData": "Недостаточно данных",
}

# burnout.basisText.*  ("На основании чего сделан вывод")
STATE_BASIS = {
    "normal": (
        "Показатели энергии, эмоционального напряжения, интонационной "
        "выразительности, темпа речи и пауз находятся в нормальном диапазоне. "
        "Если есть предыдущие записи, отрицательной динамики не выявлено."
    ),
    "shortStress": (
        "В текущей записи выявлены отдельные изменения: усиление эмоционального "
        "напряжения, изменение темпа или пауз либо снижение эмоциональной "
        "выразительности. Данных о повторяемости этих изменений пока нет."
    ),
    "sustainedStress": (
        "Изменения затрагивают несколько групп показателей и повторяются в "
        "нескольких сопоставимых записях: сохраняется напряжённая эмоциональная "
        "окраска, меняются темп и паузы, снижается положительная эмоциональная "
        "выразительность или появляются признаки снижения энергии."
    ),
    "burnoutLike": (
        "В нескольких записях сохраняется сочетание выраженных изменений: "
        "признаки снижения энергии, уменьшение эмоциональной выразительности, "
        "снижение положительной эмоциональной окраски, напряжённый эмоциональный "
        "фон и изменения темпа речи или пауз."
    ),
    "lowAffect": (
        "В речи выявлено снижение интонационного разнообразия и положительной "
        "эмоциональной окраски. Других признаков недостаточно, чтобы сделать "
        "вывод об устойчивом стрессе или выгорании."
    ),
    "insufficientData": (
        "Запись слишком короткая, содержит недостаточно речи, посторонние шумы "
        "или технические искажения. Либо для оценки динамики недостаточно "
        "предыдущих записей."
    ),
}

# burnout.conclusionText.*  ("Оценка состояния")  ← это идёт в description
STATE_CONCLUSION = {
    "normal": (
        "Значимых признаков риска не выявлено. На момент записи выраженных "
        "признаков повышенного стресса или эмоционального истощения не обнаружено."
    ),
    "shortStress": (
        "Выявлены признаки кратковременного напряжения. Результат может "
        "отражать реакцию на срочную задачу, сложный разговор, высокую нагрузку, "
        "усталость или недосып. Он не указывает на сформировавшееся выгорание."
    ),
    "sustainedStress": (
        "Выявлены признаки устойчивого напряжения. Динамика может "
        "свидетельствовать о продолжительной рабочей нагрузке и недостаточном "
        "восстановлении. Это не означает выгорание, но указывает на повышенный "
        "риск его развития."
    ),
    "burnoutLike": (
        "Выявлены признаки, совместимые с профессиональным выгоранием. "
        "Результат требует внимания, но не является диагнозом и не позволяет "
        "автоматически определить причину состояния."
    ),
    "lowAffect": (
        "Выявлена сниженная эмоциональная выразительность. Причиной могут быть "
        "усталость, смена настроения, содержание разговора, индивидуальная "
        "манера речи или условия записи. Результат не означает низкой "
        "мотивации, безразличия или нелояльности."
    ),
    "insufficientData": (
        "Недостаточно данных для надёжного вывода. Результат анализа и уровень "
        "риска не определены."
    ),
}

# burnout.comments.* — используются сервером как `state` в payload
_COMMENT_TO_STATE = {
    "need_history_to_distinguish_short_vs_chronic": "shortStress",
    "need_dynamics_for_sustained_stress": "sustainedStress",
    "burnout_compatible_not_diagnosis": "burnoutLike",
    "low_affect_nonspecific_signal": "lowAffect",

    # кириллический ключ из translations.js
    "без_истории_нескольких_текущих_записей_нельзя_надежно_различить_short_stress_vs_chronic_burnout":
        "shortStress",
}

# burnout.states ключи + алиасы сервера -> канонический ключ
_STATE_ALIASES = {
    # canonical
    "normal": "normal",
    "shortstress": "shortStress",
    "short_stress": "shortStress",
    "sustainedstress": "sustainedStress",
    "sustained_stress": "sustainedStress",
    "burnoutlike": "burnoutLike",
    "burnout_like": "burnoutLike",
    "lowaffect": "lowAffect",
    "low_affect": "lowAffect",
    "insufficientdata": "insufficientData",
    "insufficient_data": "insufficientData",

    # server comment keys
    **_COMMENT_TO_STATE,

    # generic / level-like aliases
    "low": "normal",
    "moderate": "shortStress",
    "high": "sustainedStress",
    "severe": "burnoutLike",
    "critical": "burnoutLike",
}


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
def upload_audio(api_base: str, file_path: Path, label: str = "") -> dict:
    """Upload an audio file for burnout analysis. Returns JSON with `task_id`."""
    url = f"{api_base.rstrip('/')}{UPLOAD_ENDPOINT}"
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
    """
    Poll the progress endpoint until the task completes or fails.

    The server does NOT return a `status` field. Completion is signalled by:
      - `complete: true`, OR
      - `progress >= 100` AND a non-null `result`
    Failure is signalled by a non-null `error`.
    """
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

        # ---- Terminal: failure ----
        if error:
            raise RuntimeError(f"Task failed: {error}")
        if status in ("failed", "error", "failure"):
            raise RuntimeError(
                f"Task failed: {payload.get('message') or 'unknown error'}"
            )

        # ---- Terminal: success ----
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
# Result Extraction
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


def _normalize_key(s: str) -> str:
    if not s:
        return ""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_state_key(state: str) -> str:
    """Map a raw server value to a canonical burnout-state key."""
    key = _normalize_key(state)
    if not key:
        return ""

    # direct / compact lookup
    if key in _STATE_ALIASES:
        return _STATE_ALIASES[key]
    compact = key.replace("_", "")
    if compact in _STATE_ALIASES:
        return _STATE_ALIASES[compact]

    # cyrillic keys — try raw as-is
    raw = str(state).strip()
    if raw in _COMMENT_TO_STATE:
        return _COMMENT_TO_STATE[raw]

    return ""


def _looks_like_internal_key(s: str) -> bool:
    """True if the string looks like an internal identifier, not prose."""
    if not s:
        return False
    s = s.strip()
    if " " in s:
        return False
    if any(ch.isupper() for ch in s):
        return False
    if any("\u0400" <= ch <= "\u04FF" for ch in s):
        return False
    return bool(s) and all(ch.isalnum() or ch == "_" for ch in s)


def extract_description(data: dict) -> str:
    """
    Return the Russian 'Оценка состояния' (burnout.conclusionText.*)
    for the state reported by the server.

    Order:
      1. Map `state` (and friends) via _STATE_ALIASES -> canonical key.
      2. Return STATE_CONCLUSION[canonical].
      3. Trust explicit prose fields only if they don't look like internal keys.
      4. Fallback.
    """
    if not data:
        return "Нет данных для описания."

    # 1. Resolve state.
    canon = ""
    for field in ("state", "level", "status", "risk", "verdict"):
        raw = data.get(field)
        if not raw:
            continue
        canon = _resolve_state_key(str(raw))
        if canon:
            break

    # 2. Preferred: conclusion text.
    if canon and canon in STATE_CONCLUSION:
        return STATE_CONCLUSION[canon]

    # 3. Explicit prose field, if not an internal key.
    for key in ("conclusion", "description", "comment", "summary", "message"):
        val = data.get(key)
        if isinstance(val, str) and val.strip() and not _looks_like_internal_key(val):
            return val.strip()

    # 4. Fallback summary.
    parts = []
    if data.get("state"):
        parts.append(f"Состояние: {data['state']}")
    if data.get("level"):
        parts.append(f"Уровень: {data['level']}")
    if data.get("score") is not None:
        parts.append(f"Оценка: {data['score']}")
    return "; ".join(parts) if parts else "Нет данных для описания."


def extract_state_name(data: dict) -> str:
    """Return the human-readable state name (burnout.states.*)."""
    for field in ("state", "level", "status"):
        raw = data.get(field)
        if not raw:
            continue
        canon = _resolve_state_key(str(raw))
        if canon and canon in STATE_NAMES:
            return STATE_NAMES[canon]
    return ""


def extract_basis(data: dict) -> str:
    """Return the 'Основание вывода' (burnout.basisText.*)."""
    for field in ("state", "level", "status"):
        raw = data.get(field)
        if not raw:
            continue
        canon = _resolve_state_key(str(raw))
        if canon and canon in STATE_BASIS:
            return STATE_BASIS[canon]
    return ""


# ============================================================
# Result Printing
# ============================================================
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

    state_name = extract_state_name(data)
    conclusion = extract_description(data)
    basis = extract_basis(data)

    print_section("Burnout Analysis Result")
    print(f"  State       : {state}")
    if state_name:
        print(f"  Состояние   : {state_name}")
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

    if conclusion:
        print_section("Оценка состояния")
        print(f"  {conclusion}")

    if basis:
        print_section("На основании чего сделан вывод")
        print(f"  {basis}")

    if components:
        print_section("Components")
        sorted_items = sorted(components.items(), key=lambda kv: kv[1] or 0, reverse=True)

        max_key_len = max(len(k) for k, _ in sorted_items)
        for key, value in sorted_items:
            try:
                pct = float(value) * 100 if value <= 1 else float(value)
            except (TypeError, ValueError):
                pct = 0.0

            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            label = key.replace("_", " ").title()
            print(f"  {label:<{max_key_len + 5}} {pct:6.2f}%  {bar}")

    if recommendations:
        print_section("Recommendations")
        for i, rec in enumerate(recommendations, 1):
            if isinstance(rec, dict):
                rec = rec.get("text") or rec.get("message") or json.dumps(rec)
            print(f"  {i}. {rec}")


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
# Result File Writer (TSV)
# ============================================================
class ResultFileWriter:
    """Thread-safe append-only TSV writer for filename/verdict/description rows.

    Uses a TAB separator. Fields containing tabs, newlines, or quotes are
    automatically quoted by csv.DictWriter.
    """

    FIELDNAMES = ["filename", "verdict", "description"]

    def __init__(self, path: Path):
        self.path = path
        # Write header first (truncate any existing file).
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=self.FIELDNAMES, delimiter="\t",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
        self._lock = Lock()

    def write(self, filename: str, verdict: str, description: str) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self.FIELDNAMES, delimiter="\t",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writerow({
                    "filename": filename or "",
                    "verdict": verdict or "",
                    "description": description or "",
                })
                fh.flush()


# ============================================================
# Core pipeline (single file)
# ============================================================
def process_one(
    api_base: str,
    file_path: Path,
    label: str = "",
    max_attempts: int = MAX_POLL_ATTEMPTS,
) -> tuple:
    """
    Full upload + poll + extract for a single file.
    Returns (level, description). On error, level starts with 'ERROR:'.
    """
    try:
        validate_file(file_path)
    except (FileNotFoundError, ValueError) as exc:
        return (f"ERROR:{exc}", "")

    try:
        upload_response = upload_audio(api_base, file_path, label=label)
    except RuntimeError as exc:
        return (f"ERROR:{exc}", "")

    task_id = upload_response.get("task_id") or upload_response.get("taskId")
    if not task_id:
        return ("ERROR:no task_id", "")

    try:
        final_payload = poll_until_complete(
            api_base, task_id, label=label, max_attempts=max_attempts
        )
    except (RuntimeError, TimeoutError) as exc:
        return (f"ERROR:{exc}", "")

    burnout_data = extract_burnout_data(final_payload)
    level = (burnout_data.get("level") or "unknown").strip()
    description = extract_description(burnout_data)
    return (level, description)


# ============================================================
# Batch Driver
# ============================================================
def run_batch(
    api_base: str,
    folder: Path,
    jobs: int = 1,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    tags_only: bool = False,
    quiet: bool = False,
    result_file: Path = None,
) -> int:
    """
    Process every supported audio file in `folder`.
    stdout -> result lines only.
    stderr -> human chatter (or nothing if quiet=True).
    """
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

    print(f"[i] Batch: {len(files)} file(s) in {folder} (jobs={jobs})")

    writer = ResultFileWriter(result_file) if result_file else None
    if writer:
        print(f"[i] Writing results to: {writer.path}")

    def _emit(fname: str, tag: str, description: str = "") -> None:
        if tags_only:
            sys.stdout.write(f"{tag}\n")
        else:
            sys.stdout.write(f"{fname}\t{tag}\n")
        sys.stdout.flush()

        if writer:
            writer.write(fname, tag, description)

    if jobs <= 1:
        for i, f in enumerate(files, 1):
            label = "" if quiet else f"{i}/{len(files)} {f.name}"
            tag, description = process_one(
                api_base, f, label=label, max_attempts=max_attempts
            )
            _emit(f.name, tag, description)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        labeled = [
            (("" if quiet else f"{i}/{len(files)} {f.name}"), f)
            for i, f in enumerate(files, 1)
        ]
        results = {}

        def _work(label: str, path: Path) -> tuple:
            return path, process_one(api_base, path, label=label,
                                     max_attempts=max_attempts)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_work, lbl, f): f for lbl, f in labeled}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    path, result = fut.result()
                    results[path] = result
                except Exception as exc:
                    results[f] = (f"ERROR:{exc}", "")

        for f in files:
            tag, description = results.get(f, ("ERROR", ""))
            _emit(f.name, tag, description)

    return 0


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an audio file for burnout analysis and print the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file", type=Path, nargs="?",
        help="Path to the audio file. Optional when using --batch.",
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help="Base URL of the API (e.g. https://razuma.tech/api).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full raw JSON response at the end.",
    )
    parser.add_argument(
        "--emotions", action="store_true",
        help="Also print the emotion summary.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional path to save the full JSON result to a file.",
    )
    parser.add_argument(
        "--tag", action="store_true",
        help="Print only the final burnout level to stdout and exit.",
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
        help="Suppress ALL non-result output (banners, progress, heartbeats, errors).",
    )
    parser.add_argument(
        "--tags-only", action="store_true",
        help="In --batch mode, print only the level per file (no filename).",
    )
    parser.add_argument(
        "--result-file", type=Path, metavar="PATH",
        help="Write 'filename<TAB>verdict<TAB>description' TSV rows to PATH.",
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
            args.jobs,
            max_attempts,
            tags_only=args.tags_only,
            quiet=args.quiet,
            result_file=args.result_file,
        )

    # ── Single-file mode: in --tag mode, chatter -> stderr ─
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

    print_banner("Burnout Audio Analysis")

    try:
        validate_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    # ---- Upload ----
    try:
        upload_response = upload_audio(args.api_base, args.file)
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

    # ---- Report ----
    burnout_data = extract_burnout_data(final_payload)
    description = extract_description(burnout_data)

    if args.tag:
        level = (burnout_data.get("level") or "unknown").strip()
        sys.stdout.write(level + "\n")
        return 0

    print_burnout_report(burnout_data)

    if args.emotions:
        emotion_data = extract_emotion_data(final_payload)
        print_emotion_report(emotion_data)

    if args.json:
        print_full_json(final_payload)

    if args.output:
        args.output.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[i] Full JSON saved to: {args.output}")

    if args.result_file:
        writer = ResultFileWriter(args.result_file)
        level = (burnout_data.get("level") or "unknown").strip()
        writer.write(args.file.name, level, description)
        print(f"\n[i] Result written to: {args.result_file}")

    print_banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())