# logging_utils.py
from __future__ import annotations

import os
import json
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Union

LOG_DIR = "logs"
DEFAULT_FILENAME = "training_log.json"     # array-of-objects
DEFAULT_FILENAME_JSONL = "training_log.jsonl"  # one-JSON-object-per-line


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(obj: Any, path: str) -> None:
    """
    Write JSON atomically to avoid corruption (e.g., during crashes).
    """
    dirpath = os.path.dirname(path) or "."
    _ensure_dir(dirpath)
    fd, tmppath = tempfile.mkstemp(dir=dirpath, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmppath, path)
    finally:
        # If something went wrong before replace, best-effort cleanup:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except OSError:
                pass


def _append_jsonl(record: Mapping[str, Any], path: str) -> None:
    """
    Append a single JSON object as one line (JSON Lines format).
    Creates the file if it doesn't exist.
    """
    dirpath = os.path.dirname(path) or "."
    _ensure_dir(dirpath)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def log_metrics(
    metrics: Mapping[str, Any],
    epoch: int,
    filename: str = DEFAULT_FILENAME,
    *,
    mode: str = "json",  # "json" (array) or "jsonl" (one object per line)
) -> None:
    """
    Log metrics for an epoch.

    Args:
        metrics: Dict-like metrics for this epoch (e.g., {"loss": 0.12, "acc": 0.95}).
        epoch: Epoch number (int).
        filename: Target file name inside LOG_DIR.
        mode: "json" -> keeps an array of entries (reads/writes whole file).
              "jsonl" -> appends one JSON object per line (scales better).

    Notes:
        - Writes are atomic in "json" mode.
        - "jsonl" mode is recommended for long runs.
    """
    entry = {"epoch": int(epoch), **dict(metrics)}

    if mode not in {"json", "jsonl"}:
        raise ValueError('mode must be either "json" or "jsonl"')

    path = os.path.join(LOG_DIR, filename)

    if mode == "jsonl":
        # Use .jsonl by default if caller didn't change filename.
        if filename == DEFAULT_FILENAME:
            path = os.path.join(LOG_DIR, DEFAULT_FILENAME_JSONL)
        _append_jsonl(entry, path)
    else:
        # JSON array mode (backward compatible with your original file format)
        data: List[Dict[str, Any]]
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                # If file is malformed, start fresh rather than crashing
                data = []
        else:
            data = []
        data.append(entry)
        _atomic_write_json(data, path)

    print(f"Epoch {epoch} metrics logged -> {path}")


# --------- Optional helpers (handy but not required) ---------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL metrics file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip bad lines rather than fail the whole read
                continue
    return records


def last_epoch_logged(path: str) -> int:
    """Return the last epoch number found in either JSON or JSONL file, or -1 if none."""
    if not os.path.exists(path):
        return -1
    if path.endswith(".jsonl"):
        items = read_jsonl(path)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            return -1
    if not items:
        return -1
    try:
        return int(items[-1].get("epoch", -1))
    except Exception:
        return -1
