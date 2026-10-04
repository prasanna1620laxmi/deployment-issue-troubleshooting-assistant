"""report_generator.py - builds the final report dict and renders it as Markdown.

build_report() combines:
  * facts from the parser (always deterministic),
  * the category from the classifier,
  * AI-written fields from Gemini (optional) or rule-based fallback text from the knowledge base.
"""
import datetime

import config
import knowledge_base

SEVERITY_BY_CATEGORY = {e["category"]: e.get("severity", "Medium") for e in knowledge_base.load_kb()}


def estimate_severity(category: str, parsed: dict):
    """Rule-based severity and its rationale."""
    if parsed["error_count"] == 0:
        return "Low", "No error entries were found; only warnings or informational lines."
    base = SEVERITY_BY_CATEGORY.get(category, "Medium")
    reasons = [f"category '{category}' is normally rated {base}"]
    severity = base
    if parsed["level_counts"].get("FATAL") or parsed["level_counts"].get("CRITICAL"):
        severity = "High"
        reasons.append("FATAL/CRITICAL entries are present")
    elif parsed["error_count"] >= 20 and severity == "Medium":
        severity = "High"
        reasons.append(f"{parsed['error_count']} error entries were found")
    text = "; ".join(reasons)
    return severity, text[0].upper() + text[1:] + "."


def _evidence_lines(parsed: dict, limit: int = 8) -> list:
    lines = []
    for e in parsed["errors"][:limit]:
        stamp = f"[{e['timestamp']}] " if e["timestamp"] else ""
        repeat = f" (repeated {e['count']}x)" if e["count"] > 1 else ""
        lines.append(f"{stamp}{e['message'][:300]}{repeat}")
        # also show the first few detail lines (exception text, "Caused by"), skipping stack frames
        details = [x.strip() for x in e["extra"] if not x.strip().startswith(("at ", 'File "'))]
        for detail in details[:3]:
            lines.append("    " + detail[:300])
    if not lines:
        for e in parsed["warnings"][:limit]:
            lines.append(f"[warning] {e['message'][:300]}")
    return lines


def build_report(file_name, parsed, classification, kb_entries, ai_fields=None, warnings=None,
                 tools_used=None, redactions=0, seconds=0.0) -> dict:
    """Assemble the final report. ai_fields is the validated dict from Gemini, or None."""
    warnings = list(warnings or [])
    category = classification["category"]
    severity, severity_why = estimate_severity(category, parsed)
    evidence = _evidence_lines(parsed)
    kb = kb_entries[0] if kb_entries else None
    ai = ai_fields or {}

    # --- fallback text from the knowledge base ---
    if kb:
        fb_cause = f"Probable (unconfirmed): {kb['probable_causes'][0]}"
        fb_alternatives = kb["probable_causes"][1:]
        fb_steps = kb["diagnostic_checks"] + kb["corrective_actions"]
        fb_validation = kb["validation_steps"]
    else:
        fb_cause = "No known pattern matched, so no probable cause can be suggested yet."
        fb_alternatives, fb_steps, fb_validation = [], [
            "Review the first error in the log and its surrounding lines.",
            "Collect the application logs for the failing service and the deployment pipeline log.",
            "Escalate to the application owner with the evidence shown in this report.",
        ], ["Re-run the deployment in a test environment after the cause is identified."]
        warnings.append("The error did not match any supported category; treat this report with caution.")

    steps = list(ai.get("recommended_steps") or [])
    if steps:  # second safety check, even if the agent already validated
        result = knowledge_base.validate_recommendations(steps, kb_entries, " ".join(evidence))
        steps = result["approved"]
        if result["rejected"]:
            warnings.append(f"{len(result['rejected'])} AI recommendation(s) were removed by the safety validator.")
    if not steps:
        steps = fb_steps
        if ai:
            warnings.append("AI steps were missing or rejected; knowledge-base steps are shown instead.")

    if classification.get("low_confidence"):
        warnings.append(f"The classifier confidence is low ({classification['confidence']}); best guess was "
                        f"'{classification.get('best_guess')}'.")
    if redactions:
        warnings.append(f"{redactions} sensitive value(s) were masked before analysis. Redaction is best-effort only.")

    source = "Gemini agent + knowledge base" if ai else "Rule-based fallback (no AI used)"
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "file_name": file_name,
        "source": source,
        "category": category,
        "confidence": classification.get("confidence", 0.0),
        "severity": severity,
        "severity_rationale": ai.get("severity_rationale") or severity_why,
        "executive_summary": ai.get("executive_summary") or (
            f"The log contains {parsed['error_count']} error entries and {parsed['warning_count']} warnings. "
            f"The most likely category is '{category}'. This is an automated suggestion that needs human review."),
        "evidence": evidence,
        "observed_facts": [
            f"Entries: {parsed['total_entries']}; errors: {parsed['error_count']}; warnings: {parsed['warning_count']}.",
            f"Time range: {parsed['first_timestamp'] or 'unknown'} to {parsed['last_timestamp'] or 'unknown'}.",
        ],
        "probable_root_cause": ai.get("probable_root_cause") or fb_cause,
        "alternative_causes": ai.get("alternative_causes") or fb_alternatives,
        "recommended_steps": steps,
        "validation_steps": ai.get("validation_steps") or fb_validation,
        "confidence_explanation": ai.get("confidence_explanation") or (
            f"Classifier confidence is {classification.get('confidence', 0.0)}. The model is trained on synthetic data, "
            "so treat this as a lead to verify, not a confirmed diagnosis."),
        "limitations": ai.get("limitations") or [
            "Analysis is based only on the uploaded log; other logs or metrics may change the conclusion.",
            "The classifier was trained on a small synthetic dataset."],
        "warnings": warnings,
        "tools_used": tools_used or [],
        "analysis_seconds": round(seconds, 2),
    }


def _bullets(items):
    return "\n".join(f"- {i}" for i in items) if items else "- None"


def _numbered(items):
    return "\n".join(f"{n}. {i}" for n, i in enumerate(items, 1)) if items else "- None"


def render_markdown(r: dict) -> str:
    """Turn a report dict into a Markdown document for download."""
    return f"""# Deployment Troubleshooting Report

- **File:** {r['file_name']}
- **Generated:** {r['generated_at']}
- **Method:** {r['source']}

## 1. Executive summary
{r['executive_summary']}

## 2. Detected error category
{r['category']} (classifier confidence: {r['confidence']})

## 3. Severity
**{r['severity']}** - {r['severity_rationale']}

## 4. Evidence from the log (observed facts)
{_bullets(r['observed_facts'])}

Key log lines:
```
{chr(10).join(r['evidence']) or 'No error lines found.'}
```

## 5. Probable root cause (hypothesis, not confirmed)
{r['probable_root_cause']}

## 6. Alternative possible causes
{_bullets(r['alternative_causes'])}

## 7. Recommended troubleshooting steps (priority order)
{_numbered(r['recommended_steps'])}

## 8. Validation steps
{chr(10).join('- [ ] ' + v for v in r['validation_steps'])}

## 9. Confidence / uncertainty
{r['confidence_explanation']}

## 10. Limitations and additional information needed
{_bullets(r['limitations'])}

## Warnings
{_bullets(r['warnings'])}
"""
