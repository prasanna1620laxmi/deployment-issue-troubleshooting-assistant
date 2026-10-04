import pytest

import log_parser as lp

SAMPLE = """2025-03-14 10:00:01 INFO Starting deploy
2025-03-14 10:00:05 WARN Slow registry
2025-03-14 10:00:09 ERROR Connection failed password=Hunter2
Traceback (most recent call last):
  File "app.py", line 3, in <module>
psycopg2.OperationalError: connection refused
2025-03-14 10:00:10 ERROR Connection failed password=Hunter2
"""


def test_redaction_masks_password():
    cleaned, count = lp.redact_sensitive("login password=Hunter2 and Bearer abc.def.ghi")
    assert "Hunter2" not in cleaned and "abc.def.ghi" not in cleaned
    assert count == 2


def test_redaction_masks_url_credentials():
    cleaned, _ = lp.redact_sensitive("jdbc:postgresql://user:pa55@host/db")
    assert "pa55" not in cleaned


def test_multiline_exception_is_kept_together_and_duplicates_collapsed():
    parsed = lp.parse_log(SAMPLE)
    assert parsed["error_count"] == 2
    assert len(parsed["errors"]) == 1 and parsed["errors"][0]["count"] == 2
    assert any("psycopg2" in line for line in parsed["errors"][0]["extra"])
    assert parsed["stack_traces"]
    assert parsed["warning_count"] == 1


def test_timestamps_extracted():
    parsed = lp.parse_log(SAMPLE)
    assert parsed["first_timestamp"] == "2025-03-14 10:00:01"
    assert parsed["last_timestamp"] == "2025-03-14 10:00:10"


def test_empty_and_garbage_input_do_not_crash():
    assert lp.parse_log("")["total_entries"] == 0
    assert lp.parse_log("\n\n   \n")["error_count"] == 0
    assert lp.parse_log("%%%% ???")["total_entries"] == 1


def test_read_rejects_bad_files():
    with pytest.raises(lp.LogValidationError):
        lp.read_log_file("a.exe", b"data")
    with pytest.raises(lp.LogValidationError):
        lp.read_log_file("a.log", b"   ")
    with pytest.raises(lp.LogValidationError):
        lp.read_log_file("a.log", b"abc\x00def")


def test_read_csv_with_message_column():
    text, _ = lp.read_log_file("a.csv", b"timestamp,level,message\n2025-01-01 10:00:00,error,boom\n")
    assert "ERROR boom" in text


def test_latin1_file_gives_warning():
    text, warnings = lp.read_log_file("a.log", "ERROR caf\xe9 failed".encode("latin-1"))
    assert "failed" in text and warnings
