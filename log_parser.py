"""log_parser.py - read, validate, redact and parse deployment logs.

Main functions:
    read_log_file(filename, data)  -> (text, warnings)   validates an upload
    redact_sensitive(text)         -> (text, count)      masks common secrets
    parse_log(text)                -> dict               structured summary
    focus_text(parsed)             -> str                short text for the classifier
    summarize_for_prompt(parsed)   -> str                short text for Gemini
"""
import io
import logging
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


class LogValidationError(Exception):
    """Raised when an uploaded file is not acceptable."""


# ---------------------------------------------------------------- reading
def read_log_file(filename: str, data: bytes):
    """Validate an uploaded file and return (text, warnings)."""
    warnings = []
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise LogValidationError(f"Unsupported file type '{ext}'. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}")
    if not data or not data.strip():
        raise LogValidationError("The file is empty.")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise LogValidationError(f"The file is larger than {config.MAX_FILE_MB} MB.")
    if b"\x00" in data[:4096]:
        raise LogValidationError("The file looks like a binary file, not a text log.")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        warnings.append("File is not valid UTF-8; it was read as Latin-1, so some characters may look wrong.")

    if ext == ".csv":
        text, csv_warning = _csv_to_text(text)
        if csv_warning:
            warnings.append(csv_warning)
    return text, warnings


def _csv_to_text(text: str):
    """If the CSV has a message/log column, turn rows into log lines. Otherwise keep raw text."""
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    except Exception:  # noqa: BLE001
        return text, "CSV could not be parsed as a table; it was treated as plain text."
    cols = {str(c).lower().strip(): c for c in df.columns}
    msg_col = next((cols[n] for n in ("message", "log", "text", "msg", "description") if n in cols), None)
    if msg_col is None:
        return text, "No message/log column found in the CSV; it was treated as plain text."
    ts_col = next((cols[n] for n in ("timestamp", "time", "date", "datetime") if n in cols), None)
    lvl_col = next((cols[n] for n in ("level", "severity", "loglevel") if n in cols), None)
    lines = []
    for _, row in df.iterrows():
        parts = [row[ts_col] if ts_col else "", str(row[lvl_col]).upper() if lvl_col else "", row[msg_col]]
        lines.append(" ".join(p for p in parts if p).strip())
    return "\n".join(lines), None


# ---------------------------------------------------------------- redaction
_REDACTION_RULES = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-~+/=]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "[REDACTED GOOGLE KEY]"),
    (re.compile(r"(\b\w+://[^:/\s@]+:)([^@\s]+)(@)"), r"\1[REDACTED]\3"),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|client[_-]?secret)"
            r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
]


def redact_sensitive(text: str):
    """Mask common secrets. This is best-effort only: it does NOT catch every secret."""
    total = 0
    for pattern, replacement in _REDACTION_RULES:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


# ---------------------------------------------------------------- parsing
TIMESTAMP_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?")
LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|FATAL|CRITICAL|SEVERE)\b")
ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}
LEVEL_ALIASES = {"WARN": "WARNING", "SEVERE": "ERROR"}
ERROR_WORDS = re.compile(
    r"(?i)\b(exception|traceback|failed|failure|denied|refused|timed out|timeout|cannot|can't|unable|fatal|"
    r"not found|no such|out of memory|oomkilled|forbidden|unauthorized)\b"
)
STACK_RE = re.compile(r"(Traceback \(most recent call last\)|^\s+at\s+\S+|^\s*Caused by:|^\s+File \".*\", line \d+)", re.M)


def _split_entries(text: str):
    """Group lines into entries. Lines without a timestamp/level join the previous entry."""
    entries = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        ts_match = TIMESTAMP_RE.match(line)
        lvl_match = LEVEL_RE.search(line[:50])
        if ts_match or lvl_match:
            level = lvl_match.group(1) if lvl_match else "UNKNOWN"
            level = LEVEL_ALIASES.get(level, level)
            message = line
            if ts_match:
                message = line[ts_match.end():]
            if lvl_match and lvl_match.start() < 50:
                message = LEVEL_RE.sub("", message, count=1)
            message = message.strip(" []-:\t")
            entries.append(
                {"line_no": line_no, "timestamp": ts_match.group(1) if ts_match else None,
                 "level": level, "message": message or line.strip(), "extra": []}
            )
        elif entries:
            entries[-1]["extra"].append(line.rstrip())
        else:
            entries.append({"line_no": line_no, "timestamp": None, "level": "UNKNOWN", "message": line.strip(), "extra": []})
    for e in entries:
        full = e["message"] + "\n" + "\n".join(e["extra"])
        e["is_error"] = e["level"] in ERROR_LEVELS or (e["level"] == "UNKNOWN" and bool(ERROR_WORDS.search(full)))
        e["has_stack"] = bool(STACK_RE.search(full)) if e["extra"] else bool(STACK_RE.search(e["message"]))
    return entries


def _collapse(entries):
    """Merge repeated identical messages (digits ignored) and keep a count."""
    unique = OrderedDict()
    for e in entries:
        key = (e["level"], re.sub(r"\d+", "#", e["message"]))
        if key in unique:
            unique[key]["count"] += 1
        else:
            unique[key] = dict(e, count=1)
    return list(unique.values())


def parse_log(text: str) -> dict:
    """Return a structured summary of the log. Never raises on odd input."""
    text = text or ""
    entries = _split_entries(text)
    level_counts = {}
    for e in entries:
        level_counts[e["level"]] = level_counts.get(e["level"], 0) + 1
    errors = [e for e in entries if e["is_error"]]
    warnings = [e for e in entries if e["level"] == "WARNING"]
    stacks = []
    for e in entries:
        if e["has_stack"] and len(stacks) < 5:
            stacks.append("\n".join([e["message"]] + e["extra"][:15]))
    timestamps = [e["timestamp"] for e in entries if e["timestamp"]]
    return {
        "total_lines": len(text.splitlines()),
        "total_entries": len(entries),
        "level_counts": level_counts,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "errors": _collapse(errors)[:15],
        "warnings": _collapse(warnings)[:10],
        "stack_traces": stacks,
    }


def focus_text(parsed: dict, max_chars: int = 1500) -> str:
    """Short text containing the most important lines. This is what the classifier reads."""
    source = parsed["errors"] or parsed["warnings"]
    parts = []
    for e in source[:6]:
        parts.append(" ".join([e["message"]] + [x.strip() for x in e["extra"][:2]]))
    return " ".join(parts)[:max_chars]


def summarize_for_prompt(parsed: dict, max_chars: int = config.MAX_AI_CHARS) -> str:
    """Compact, sanitized summary for the AI agent (keeps token usage low)."""
    lines = [
        f"Entries: {parsed['total_entries']}, errors: {parsed['error_count']}, warnings: {parsed['warning_count']}",
        f"Time range: {parsed['first_timestamp']} -> {parsed['last_timestamp']}",
        f"Level counts: {parsed['level_counts']}",
        "Key errors:",
    ]
    for e in parsed["errors"][:8]:
        lines.append(f"- [{e['timestamp'] or 'no-ts'}] {e['message'][:300]} (x{e['count']})")
    lines.append("Warnings:")
    for e in parsed["warnings"][:5]:
        lines.append(f"- {e['message'][:200]} (x{e['count']})")
    lines.append("Stack traces (truncated):")
    for s in parsed["stack_traces"][:2]:
        lines.append(s[:600])
    return "\n".join(lines)[:max_chars]
