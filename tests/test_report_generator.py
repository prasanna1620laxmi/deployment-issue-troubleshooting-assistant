import knowledge_base as kb
import log_parser as lp
import report_generator as rg

LOG = "2025-03-14 10:00:09 ERROR java.lang.OutOfMemoryError: Java heap space\n"
CLASSIFICATION = {"category": "Resource Exhaustion", "best_guess": "Resource Exhaustion", "confidence": 0.9,
                  "low_confidence": False, "probabilities": {}}


def _report(ai_fields=None, classification=CLASSIFICATION):
    parsed = lp.parse_log(LOG)
    entries = kb.search_kb(classification["category"])
    return rg.build_report("x.log", parsed, classification, entries, ai_fields)


def test_fallback_report_has_all_sections():
    md = rg.render_markdown(_report())
    for heading in ["Executive summary", "Detected error category", "Severity", "Probable root cause",
                    "Recommended troubleshooting steps", "Validation steps", "Limitations"]:
        assert heading in md


def test_root_cause_is_worded_as_unconfirmed():
    assert "unconfirmed" in _report()["probable_root_cause"].lower()


def test_dangerous_ai_step_is_removed():
    ai = {"recommended_steps": ["Increase the memory limit through the change process", "Run rm -rf /tmp/cache"]}
    report = _report(ai)
    assert all("rm -rf" not in s for s in report["recommended_steps"])
    assert any("removed by the safety validator" in w for w in report["warnings"])


def test_unknown_category_adds_warning():
    unknown = {"category": "Unknown / Unsupported", "best_guess": None, "confidence": 0.1,
               "low_confidence": True, "probabilities": {}}
    report = _report(classification=unknown)
    assert report["warnings"] and report["recommended_steps"]


def test_no_errors_gives_low_severity():
    parsed = lp.parse_log("2025-01-01 10:00:00 INFO all good")
    assert rg.estimate_severity("Build Failure", parsed)[0] == "Low"
