import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.models import Market
from poly_alpha_lab.resolution_analyzer import analyze_resolution


LOW_RISK_RULES = (
    "This market will resolve to Yes if the official NOAA website reports a temperature "
    "below 24 by 11:59 PM ET on May 9. Otherwise, this market resolves to No."
)


def market(**overrides: object) -> Market:
    data = {
        "id": "m1",
        "question": "Will test resolve clearly?",
        "slug": "test-resolve-clearly",
        "category": "Weather",
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.4, 0.6],
        "clobTokenIds": ["yes-token", "no-token"],
    }
    data.update(overrides)
    return Market.model_validate(data)


def test_resolution_criteria_extraction() -> None:
    analysis = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))

    assert analysis.has_resolution_text
    assert analysis.resolution_text_source == 'market.raw["resolutionCriteria"]'
    assert "official NOAA" in analysis.resolution_text_excerpt


def test_rules_extraction() -> None:
    analysis = analyze_resolution(market(rules=LOW_RISK_RULES))

    assert analysis.resolution_text_source == 'market.raw["rules"]'


def test_description_fallback_extraction() -> None:
    analysis = analyze_resolution(market(description=LOW_RISK_RULES))

    assert analysis.resolution_text_source == 'market.raw["description"]'


def test_question_only_warning_and_missing_resolution_text() -> None:
    analysis = analyze_resolution(market(question="Will this happen?"))

    assert not analysis.has_resolution_text
    assert analysis.resolution_text_source == "market.question"
    assert "question_only_no_resolution_text" in analysis.warnings
    assert "resolution_text" in analysis.missing_fields


def test_official_source_deadline_timezone_is_low_risk() -> None:
    analysis = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))

    assert analysis.ambiguity_risk == "LOW"
    assert analysis.dispute_risk == "LOW"
    assert analysis.risk_score < 30


def test_missing_source_of_truth_increases_risk() -> None:
    with_source = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))
    no_source = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if the temperature is below 24 by "
                "11:59 PM ET on May 9. Otherwise, this market resolves to No."
            )
        )
    )

    assert "resolution_source" in no_source.missing_fields
    assert no_source.risk_score > with_source.risk_score


def test_missing_deadline_increases_risk() -> None:
    with_deadline = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))
    no_deadline = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if the official NOAA website reports "
                "a temperature below 24. Otherwise, this market resolves to No."
            )
        )
    )

    assert "deadline" in no_deadline.missing_fields
    assert no_deadline.risk_score > with_deadline.risk_score


def test_subjective_keywords_increase_risk() -> None:
    objective = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))
    subjective = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if credible reports indicate a significant "
                "and widely recognized event by 11:59 PM ET. Otherwise, resolves to No."
            )
        )
    )

    assert subjective.risk_score > objective.risk_score
    assert subjective.dispute_risk in {"MEDIUM", "HIGH"}


def test_preliminary_revised_triggers_warning() -> None:
    analysis = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if the official BLS preliminary or revised "
                "CPI data is above 3 by 11:59 PM ET. Otherwise, resolves to No."
            )
        )
    )

    assert "preliminary_revised_data_risk" in analysis.warnings


def test_announcement_vs_implementation_triggers_warning() -> None:
    analysis = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if the law is announced and takes effect "
                "by 11:59 PM ET according to the official government website. Otherwise, "
                "resolves to No."
            )
        )
    )

    assert "announcement_vs_implementation_ambiguity" in analysis.warnings


def test_numeric_threshold_lowers_risk() -> None:
    numeric = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))
    vague = analyze_resolution(
        market(
            resolutionCriteria=(
                "This market will resolve to Yes if the official NOAA website reports a "
                "significant temperature event by 11:59 PM ET. Otherwise, resolves to No."
            )
        )
    )

    assert numeric.risk_score < vague.risk_score


def test_yes_and_no_criteria_extraction() -> None:
    analysis = analyze_resolution(market(resolutionCriteria=LOW_RISK_RULES))

    assert analysis.what_counts_as_yes is not None
    assert "resolve to Yes if" in analysis.what_counts_as_yes
    assert analysis.what_counts_as_no is not None
    assert "Otherwise" in analysis.what_counts_as_no


def test_analyze_resolution_cli_runs(monkeypatch, capsys) -> None:
    class FakeGammaClient:
        def get_market(self, market_id: str) -> Market:
            return market(id=market_id, resolutionCriteria=LOW_RISK_RULES)

        def get_market_by_slug(self, slug: str) -> Market:
            return market(slug=slug, resolutionCriteria=LOW_RISK_RULES)

    monkeypatch.setattr(cli, "GammaClient", FakeGammaClient)

    assert cli.run(["analyze-resolution", "--market-id", "m1"]) == 0
    output = capsys.readouterr().out

    assert "# Resolution Analysis" in output
    assert "risk_score" in output
