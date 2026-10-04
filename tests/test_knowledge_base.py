import config
import knowledge_base as kb


def test_every_category_has_complete_entry():
    for category in config.CATEGORIES:
        entry = kb.get_entry(category)
        assert entry is not None, category
        for key in ("probable_causes", "diagnostic_checks", "corrective_actions", "validation_steps"):
            assert entry[key], f"{category} missing {key}"


def test_keyword_search_finds_expected_category():
    assert kb.keyword_category("java.lang.OutOfMemoryError: Java heap space")[0] == "Resource Exhaustion"
    assert kb.keyword_category("ModuleNotFoundError: No module named 'flask'")[0] == "Dependency Issue"


def test_unknown_text_returns_unknown():
    assert kb.keyword_category("the cafeteria menu is attached")[0] == config.UNKNOWN_CATEGORY


def test_validator_rejects_dangerous_and_unrelated():
    kb_entries = kb.search_kb("Database Connectivity Failure")
    result = kb.validate_recommendations(
        ["Verify the database host and port", "Run rm -rf /var/app", "Buy a new keyboard"], kb_entries)
    assert result["approved"] == ["Verify the database host and port"]
    assert len(result["rejected"]) == 2
