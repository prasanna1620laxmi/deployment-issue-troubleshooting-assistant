"""knowledge_base.py - load the troubleshooting knowledge base, search it, and validate recommendations."""
import json
import re

import config

_CACHE = None

# Phrases that suggest a risky or operational action. Recommendations with these are rejected.
DANGEROUS_PATTERNS = [
    r"rm\s+-rf", r"\bdrop\s+(table|database)\b", r"\btruncate\b", r"\bdelete\s+from\b", r"chmod\s+(-R\s+)?777",
    r"\bmkfs\b", r"\bshutdown\b", r"\breboot\b", r"kill\s+-9", r"--force\b", r"\bdisable\s+(the\s+)?(firewall|security|authentication)\b",
    r"\bformat\s+(the\s+)?(disk|drive)\b", r"\bgrant\s+all\b", r"\bsudo\b",
]
STOPWORDS = {
    "check", "that", "with", "from", "your", "this", "have", "verify", "ensure", "review", "before", "after",
    "into", "when", "then", "also", "those", "they", "them", "been", "only", "each", "using", "should", "which",
    "confirm", "make", "sure", "step", "steps", "the", "and", "for",
}


def load_kb(path=config.KNOWLEDGE_BASE_JSON, force: bool = False) -> list:
    global _CACHE
    if _CACHE is None or force:
        with open(path, encoding="utf-8") as f:
            _CACHE = json.load(f)
        required = {"category", "patterns", "probable_causes", "diagnostic_checks", "corrective_actions", "validation_steps"}
        for entry in _CACHE:
            missing = required - set(entry)
            if missing:
                raise ValueError(f"Knowledge base entry '{entry.get('category')}' is missing: {missing}")
    return _CACHE


def get_entry(category: str):
    for entry in load_kb():
        if entry["category"].lower() == (category or "").lower():
            return entry
    return None


def score_text(text: str) -> list:
    """Return [(score, entry)] for entries whose patterns appear in text (best first)."""
    lowered = (text or "").lower()
    scored = []
    for entry in load_kb():
        score = sum(1 for p in entry["patterns"] if p in lowered)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def keyword_category(text: str):
    """Rule-based category guess used when the TensorFlow model is unavailable. Returns (category, score)."""
    scored = score_text(text)
    if not scored:
        return config.UNKNOWN_CATEGORY, 0
    return scored[0][1]["category"], scored[0][0]


def search_kb(query: str, top_k: int = 2) -> list:
    """Find entries by exact category name first, then by keyword patterns."""
    entry = get_entry(query)
    if entry:
        return [dict(entry, match_score=99)]
    return [dict(e, match_score=s) for s, e in score_text(query)[:top_k]]


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOPWORDS}


def validate_recommendations(recommendations: list, kb_entries: list, evidence_text: str = "") -> dict:
    """Keep only recommendations that are safe and related to the evidence or knowledge base."""
    allowed_words = _words(evidence_text)
    for entry in kb_entries:
        allowed_words |= _words(" ".join([entry["category"]] + entry["probable_causes"] + entry["diagnostic_checks"]
                                         + entry["corrective_actions"] + entry["validation_steps"]))
    approved, rejected = [], []
    for rec in recommendations:
        rec = str(rec).strip()
        if not rec:
            continue
        if any(re.search(p, rec, re.I) for p in DANGEROUS_PATTERNS):
            rejected.append({"text": rec, "reason": "Contains a risky or operational action."})
        elif not (_words(rec) & allowed_words):
            rejected.append({"text": rec, "reason": "Not related to the evidence or known procedures."})
        else:
            approved.append(rec)
    return {"approved": approved, "rejected": rejected}
