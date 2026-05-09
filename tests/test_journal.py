import csv
import json

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.journal import EXPORT_FIELDS, ResearchJournal


def make_journal(tmp_path):
    return ResearchJournal(tmp_path / "journal.sqlite")


def add_entry(journal: ResearchJournal, **overrides):
    data = {
        "market_id": "m1",
        "slug": "test-market",
        "question": "Test market?",
        "category": "economics",
        "candidate_score": 80.0,
        "candidate_grade": "A",
        "side": "YES",
        "fair_yes_probability": 0.6,
        "probability_source": "manual",
        "entry_price": 0.5,
        "entry_size": 10.0,
        "fee_per_share": 0.001,
        "rationale": "test",
    }
    data.update(overrides)
    return journal.create_entry(**data)


def test_journal_add_and_list(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal)

    entries = journal.list_entries(limit=5)

    assert entry.id == 1
    assert len(entries) == 1
    assert entries[0].market_id == "m1"
    assert entries[0].expected_value_per_share == pytest.approx(0.099)
    assert entries[0].expected_profit == pytest.approx(0.99)


def test_resolve_yes_win_pnl(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal, side="YES", entry_price=0.5, entry_size=10, fee_per_share=0.001)

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=1)

    assert resolved.realized_pnl == pytest.approx((1 - 0.5 - 0.001) * 10)


def test_resolve_yes_loss_pnl(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal, side="YES", entry_price=0.5, entry_size=10, fee_per_share=0.001)

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=0)

    assert resolved.realized_pnl == pytest.approx((0 - 0.5 - 0.001) * 10)


def test_resolve_no_win_pnl(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal, side="NO", entry_price=0.4, entry_size=10, fee_per_share=0.001)

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=0)

    assert resolved.realized_pnl == pytest.approx(((1 - 0) - 0.4 - 0.001) * 10)


def test_resolve_no_loss_pnl(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal, side="NO", entry_price=0.4, entry_size=10, fee_per_share=0.001)

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=1)

    assert resolved.realized_pnl == pytest.approx(((1 - 1) - 0.4 - 0.001) * 10)


def test_brier_score(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal, fair_yes_probability=0.6)

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=1)

    assert resolved.brier_score == pytest.approx((0.6 - 1) ** 2)


def test_side_none_has_no_pnl(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(
        journal,
        side="NONE",
        entry_price=None,
        entry_size=None,
        fee_per_share=None,
        status="SKIPPED",
    )

    resolved = journal.update_resolution(entry_id=entry.id, resolution_value=1)

    assert resolved.realized_pnl is None
    assert resolved.brier_score == pytest.approx((0.6 - 1) ** 2)


def test_summary_groups_by_category_grade_and_probability_source(tmp_path) -> None:
    journal = make_journal(tmp_path)
    yes = add_entry(journal, market_id="yes", category="economics", candidate_grade="A")
    no = add_entry(
        journal,
        market_id="no",
        category="sports",
        candidate_grade="B",
        side="NO",
        entry_price=0.4,
        probability_source="unknown",
    )
    journal.update_resolution(entry_id=yes.id, resolution_value=1)
    journal.update_resolution(entry_id=no.id, resolution_value=0)

    summary = journal.summarize_performance()

    assert summary["total_entries"] == 2
    assert summary["resolved_entries"] == 2
    assert summary["by_category"]["economics"]["entries"] == 1
    assert summary["by_category"]["sports"]["entries"] == 1
    assert summary["by_candidate_grade"]["A"]["entries"] == 1
    assert summary["by_candidate_grade"]["B"]["entries"] == 1
    assert summary["by_probability_source"]["manual"]["entries"] == 1
    assert summary["by_probability_source"]["unknown"]["entries"] == 1


def test_delete_existing_entry(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal)

    journal.delete_entry(entry.id)

    assert journal.list_entries(limit=10) == []


def test_delete_nonexistent_entry_raises_clear_error(tmp_path) -> None:
    journal = make_journal(tmp_path)

    with pytest.raises(LookupError, match="journal entry not found: 999"):
        journal.delete_entry(999)


def test_delete_updates_summary(tmp_path) -> None:
    journal = make_journal(tmp_path)
    entry = add_entry(journal)
    journal.update_resolution(entry_id=entry.id, resolution_value=1)

    journal.delete_entry(entry.id)
    summary = journal.summarize_performance()

    assert summary["total_entries"] == 0
    assert summary["resolved_entries"] == 0
    assert summary["total_realized_pnl"] == 0.0


def test_export_csv_success_and_fields(tmp_path) -> None:
    journal = make_journal(tmp_path)
    add_entry(journal, notes="hello")
    output = tmp_path / "journal_export.csv"

    row_count = journal.export_csv(output)

    assert row_count == 1
    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == EXPORT_FIELDS
    assert rows[0]["market_id"] == "m1"
    assert rows[0]["rationale"] == "test"
    assert rows[0]["notes"] == "hello"


def test_export_filters_status_category_and_grade(tmp_path) -> None:
    journal = make_journal(tmp_path)
    resolved = add_entry(journal, market_id="resolved", category="economics", candidate_grade="A")
    add_entry(journal, market_id="open-sports", category="sports", candidate_grade="A")
    add_entry(journal, market_id="open-econ-b", category="economics", candidate_grade="B")
    journal.update_resolution(entry_id=resolved.id, resolution_value=1)
    output = tmp_path / "filtered.csv"

    row_count = journal.export_csv(output, status="RESOLVED", category="economics", grade="A")

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert row_count == 1
    assert rows[0]["market_id"] == "resolved"


def test_empty_database_summary_is_stable(tmp_path) -> None:
    journal = make_journal(tmp_path)

    summary = journal.summarize_performance()

    assert summary["total_entries"] == 0
    assert summary["open_entries"] == 0
    assert summary["resolved_entries"] == 0
    assert summary["total_realized_pnl"] == 0.0
    assert summary["average_brier_score"] is None
    assert summary["average_pnl"] is None
    assert summary["by_category"] == {}


def test_cli_add_from_json_file_creates_entry(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli.sqlite"
    monkeypatch.setattr(cli.settings, "journal_db_path", str(db_path))
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            {
                "market_id": "json-1",
                "slug": "json-market",
                "question": "JSON market?",
                "category": "economics",
                "candidate_score": 82.5,
                "candidate_grade": "A",
                "side": "YES",
                "fair_yes_probability": 0.62,
                "probability_source": "manual",
                "entry_price": 0.55,
                "entry_size": 10,
                "fee_per_share": 0.001,
                "expected_value_per_share": 0.069,
                "expected_profit": 0.69,
                "rationale": "json draft",
            }
        ),
        encoding="utf-8",
    )

    assert cli.run(["journal", "add", "--from-json-file", str(draft)]) == 0
    entries = ResearchJournal(db_path).list_entries(limit=5)

    assert len(entries) == 1
    assert entries[0].market_id == "json-1"
    assert entries[0].expected_value_per_share == pytest.approx(0.069)
    assert entries[0].expected_profit == pytest.approx(0.69)


def test_cli_add_from_json_file_allows_cli_override(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli.sqlite"
    monkeypatch.setattr(cli.settings, "journal_db_path", str(db_path))
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            {
                "market_id": "json-override",
                "question": "JSON override?",
                "category": "economics",
                "side": "NONE",
                "rationale": "json draft",
            }
        ),
        encoding="utf-8",
    )

    assert cli.run(
        [
            "journal",
            "add",
            "--from-json-file",
            str(draft),
            "--category",
            "sports",
            "--side",
            "YES",
            "--fair-yes",
            "0.6",
            "--entry-price",
            "0.5",
            "--entry-size",
            "10",
            "--fee-per-share",
            "0.001",
        ]
    ) == 0
    entry = ResearchJournal(db_path).list_entries(limit=5)[0]

    assert entry.category == "sports"
    assert entry.side == "YES"
    assert entry.status == "OPEN"


def test_cli_old_journal_add_args_still_work(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli.sqlite"
    monkeypatch.setattr(cli.settings, "journal_db_path", str(db_path))

    assert cli.run(
        [
            "journal",
            "add",
            "--market-id",
            "old-1",
            "--slug",
            "old-market",
            "--question",
            "Old style?",
            "--category",
            "economics",
            "--candidate-score",
            "80",
            "--candidate-grade",
            "A",
            "--side",
            "YES",
            "--fair-yes",
            "0.6",
            "--probability-source",
            "manual",
            "--entry-price",
            "0.5",
            "--entry-size",
            "10",
            "--fee-per-share",
            "0.001",
            "--rationale",
            "old style",
        ]
    ) == 0
    entry = ResearchJournal(db_path).list_entries(limit=5)[0]

    assert entry.market_id == "old-1"
    assert entry.expected_value_per_share == pytest.approx(0.099)
