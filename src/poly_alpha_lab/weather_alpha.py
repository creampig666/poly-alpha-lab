"""Weather threshold paper alpha module."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.market_type_classifier import (
    MarketClassification,
    MarketType,
    classify_strategy_candidate,
)
from poly_alpha_lab.weather_data import WeatherDataProvider, WeatherForecast
from poly_alpha_lab.weather_probability_model import (
    BucketMode,
    WeatherModel,
    convert_temperature,
    convert_temperature_std,
    estimate_temperature_threshold_probability,
)
from poly_alpha_lab.resolution_analyzer import ResolutionAnalysis
from poly_alpha_lab.weather_calibration import (
    WeatherCalibrationSummary,
    calibration_quality_for_n,
    calibration_quality_warnings,
    current_forecast_calibration_key,
    normalize_group_by,
)

PaperSide = Literal["YES", "NO", "NONE"]
SignalStatus = Literal[
    "VALID",
    "NEEDS_MANUAL_REVIEW",
    "INVALID_DATA",
    "RULE_MISMATCH",
    "MISSING_FORECAST_DATA",
    "STALE_OR_FUTURE_FORECAST",
    "LOCATION_SOURCE_MISMATCH",
]
RiskLabel = Literal["LOW", "MEDIUM", "HIGH"]
RunMode = Literal["live", "replay"]
ResolutionLookup = Callable[[dict[str, Any]], ResolutionAnalysis | None]
CalibrationSummaries = dict[str, WeatherCalibrationSummary]

WEATHER_ALPHA_RATIONALE = (
    "weather threshold model paper signal; requires manual review; not a real trading signal"
)


class WeatherAlphaSignal(BaseModel):
    """Paper-only weather threshold signal."""

    market_id: str
    slug: str | None = None
    question: str
    category: str | None = None
    location_name: str
    metric: str
    target_date: str
    threshold: float
    unit: str
    bucket_mode: BucketMode | None = None
    bucket_lower_bound: float | None = None
    bucket_upper_bound: float | None = None
    bucket_assumption: str | None = None
    forecast_mean: float
    forecast_std: float
    model_p_yes: float = Field(ge=0, le=1)
    yes_breakeven: float
    no_upper_bound: float
    yes_model_edge: float
    no_model_edge: float
    suggested_paper_side: PaperSide
    edge_threshold: float
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    signal_status: SignalStatus = "VALID"
    validation_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
    as_of_time: datetime | None = None
    forecast_issued_at: datetime | None = None
    forecast_time_tolerance_seconds: float = 0.0
    forecast_timing_tolerance_applied: bool = False
    resolution_source: str | None = None
    resolution_risk_score: float | None = None
    ambiguity_risk: RiskLabel | None = None
    dispute_risk: RiskLabel | None = None
    source_location_name: str | None = None
    station_id: str | None = None
    forecast_station_id: str | None = None
    resolution_station_id: str | None = None
    bucket_boundary_confirmed: bool = False
    bucket_structure_confirmed: bool = False
    bucket_numeric_boundary_confirmed: bool = False
    forecast_data_source: str = "unknown"
    forecast_source: str | None = None
    forecast_model: str | None = None
    std_method: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    actual_source: str | None = None
    cache_key: str | None = None
    raw_data_reference: str | None = None
    data_provenance_warnings: list[str] = Field(default_factory=list)
    run_mode: RunMode = "live"
    weather_model: WeatherModel = "normal"
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    distribution_assumption: str = "normal forecast error"
    calibration_group_key: str | None = None
    calibration_n: int | None = None
    calibration_bias: float | None = None
    calibration_std_error: float | None = None
    calibration_applied: bool = False
    calibration_quality: str | None = None
    calibration_quality_warnings: list[str] = Field(default_factory=list)
    calibration_min_samples_required: int | None = None
    calibration_bias_raw: float | None = None
    calibration_bias_shrunk: float | None = None
    calibration_std_error_raw: float | None = None
    calibration_std_error_used: float | None = None
    forecast_mean_raw: float | None = None
    forecast_mean_adjusted: float | None = None
    forecast_std_raw: float | None = None
    forecast_std_calibrated: float | None = None
    journal_draft_payload: dict[str, Any] = Field(default_factory=dict)
    strategy_score: float | None = None


class WeatherAlphaScanResult(BaseModel):
    weather_candidate_count: int
    threshold_candidate_count: int = 0
    exact_bucket_candidate_count: int = 0
    signals: list[WeatherAlphaSignal]
    skipped: list[str] = Field(default_factory=list)


class WeatherSignalValidation(BaseModel):
    signal_status: SignalStatus = "VALID"
    validation_warnings: list[str] = Field(default_factory=list)
    resolution_source: str | None = None
    resolution_risk_score: float | None = None
    ambiguity_risk: RiskLabel | None = None
    dispute_risk: RiskLabel | None = None
    source_location_name: str | None = None
    station_id: str | None = None
    forecast_station_id: str | None = None
    resolution_station_id: str | None = None
    bucket_boundary_confirmed: bool = False
    bucket_structure_confirmed: bool = False
    bucket_numeric_boundary_confirmed: bool = False
    data_provenance_warnings: list[str] = Field(default_factory=list)
    forecast_timing_tolerance_applied: bool = False


def run_weather_alpha_scan(
    strategy_json_path: str | Path,
    provider: WeatherDataProvider,
    *,
    edge_threshold: float = 0.05,
    bucket_mode: BucketMode = "rounded",
    weather_model: WeatherModel = "normal",
    student_t_df: float = 5,
    mixture_tail_weight: float = 0.10,
    mixture_tail_scale: float = 2.5,
    as_of_time: str | datetime | None = None,
    resolution_lookup: ResolutionLookup | None = None,
    calibration_summaries: CalibrationSummaries | None = None,
    calibration_group: list[str] | str = "metric,horizon_bucket",
    use_calibrated_std: bool = False,
    use_calibrated_bias: bool = False,
    min_calibration_samples: int = 30,
    allow_low_quality_calibration: bool = False,
    forecast_time_tolerance_seconds: float = 0.0,
) -> WeatherAlphaScanResult:
    """Read strategy candidates JSON and build weather alpha signals."""

    path = Path(strategy_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("strategy JSON must contain a list of candidates")

    generated_at = datetime.now(UTC)
    parsed_as_of_time = _parse_datetime(as_of_time) if as_of_time is not None else generated_at
    defaulted_as_of_time = as_of_time is None
    weather_candidate_count = 0
    threshold_candidate_count = 0
    exact_bucket_candidate_count = 0
    signals: list[WeatherAlphaSignal] = []
    skipped: list[str] = []
    for candidate in data:
        if not isinstance(candidate, dict):
            continue
        classification = classify_strategy_candidate(candidate)
        if classification.market_type not in {
            MarketType.weather_temperature_threshold,
            MarketType.weather_temperature_exact_bucket,
        }:
            continue
        weather_candidate_count += 1
        if classification.market_type == MarketType.weather_temperature_exact_bucket:
            exact_bucket_candidate_count += 1
        else:
            threshold_candidate_count += 1
        provider_as_of_time = parsed_as_of_time if as_of_time is not None else None
        forecast = provider.get_forecast(
            classification.location_name or "",
            classification.target_date or "",
            classification.metric or "",
            as_of_time=provider_as_of_time,
        )
        if forecast is None:
            skipped.append(
                f"{candidate.get('market_id') or 'unknown'}:MISSING_FORECAST_DATA:"
                f"{classification.location_name}:{classification.target_date}:{classification.metric}"
            )
            continue
        resolution = resolution_lookup(candidate) if resolution_lookup is not None else None
        signals.append(
            build_weather_alpha_signal(
                candidate,
                classification,
                forecast,
                edge_threshold=edge_threshold,
                bucket_mode=bucket_mode,
                weather_model=weather_model,
                student_t_df=student_t_df,
                mixture_tail_weight=mixture_tail_weight,
                mixture_tail_scale=mixture_tail_scale,
                generated_at=generated_at,
                as_of_time=parsed_as_of_time,
                as_of_time_defaulted=defaulted_as_of_time,
                resolution_analysis=resolution,
                calibration_summaries=calibration_summaries,
                calibration_group=calibration_group,
                use_calibrated_std=use_calibrated_std,
                use_calibrated_bias=use_calibrated_bias,
                min_calibration_samples=min_calibration_samples,
                allow_low_quality_calibration=allow_low_quality_calibration,
                forecast_time_tolerance_seconds=forecast_time_tolerance_seconds,
            )
        )

    signals.sort(
        key=lambda signal: max(signal.yes_model_edge, signal.no_model_edge),
        reverse=True,
    )
    return WeatherAlphaScanResult(
        weather_candidate_count=weather_candidate_count,
        threshold_candidate_count=threshold_candidate_count,
        exact_bucket_candidate_count=exact_bucket_candidate_count,
        signals=signals,
        skipped=skipped,
    )


def build_weather_alpha_signal(
    candidate: dict[str, Any],
    classification: MarketClassification,
    forecast: WeatherForecast,
    *,
    edge_threshold: float = 0.05,
    bucket_mode: BucketMode = "rounded",
    weather_model: WeatherModel = "normal",
    student_t_df: float = 5,
    mixture_tail_weight: float = 0.10,
    mixture_tail_scale: float = 2.5,
    generated_at: datetime | None = None,
    as_of_time: str | datetime | None = None,
    as_of_time_defaulted: bool = False,
    resolution_analysis: ResolutionAnalysis | None = None,
    calibration_summaries: CalibrationSummaries | None = None,
    calibration_group: list[str] | str = "metric,horizon_bucket",
    use_calibrated_std: bool = False,
    use_calibrated_bias: bool = False,
    min_calibration_samples: int = 30,
    allow_low_quality_calibration: bool = False,
    forecast_time_tolerance_seconds: float = 0.0,
) -> WeatherAlphaSignal:
    """Build one signal from a classified candidate and one forecast snapshot."""

    if classification.market_type not in {
        MarketType.weather_temperature_threshold,
        MarketType.weather_temperature_exact_bucket,
    }:
        raise ValueError("classification must be a supported weather temperature market")
    if (
        classification.location_name is None
        or classification.metric is None
        or classification.target_date is None
        or classification.threshold_value is None
        or classification.unit is None
        or classification.comparator is None
    ):
        raise ValueError("classification is missing required weather threshold fields")

    generated_at = _aware_datetime(generated_at or datetime.now(UTC))
    parsed_as_of_time = _parse_datetime(as_of_time) if as_of_time is not None else generated_at
    run_mode: RunMode = (
        "live"
        if as_of_time_defaulted
        or parsed_as_of_time == generated_at
        or forecast_time_tolerance_seconds > 0
        else "replay"
    )
    calibration_result = _calibrated_forecast(
        forecast=forecast,
        classification=classification,
        calibration_summaries=calibration_summaries,
        calibration_group=calibration_group,
        use_calibrated_std=use_calibrated_std,
        use_calibrated_bias=use_calibrated_bias,
        min_calibration_samples=min_calibration_samples,
        allow_low_quality_calibration=allow_low_quality_calibration,
    )
    forecast_for_model = calibration_result["forecast"]
    forecast_issued_at, forecast_time_warning = _parse_optional_datetime(forecast.forecast_issued_at)
    probability = estimate_temperature_threshold_probability(
        forecast=forecast_for_model,
        threshold=classification.threshold_value,
        comparator=classification.comparator,
        threshold_unit=classification.unit,
        bucket_mode=bucket_mode,
        weather_model=weather_model,
        student_t_df=student_t_df,
        mixture_tail_weight=mixture_tail_weight,
        mixture_tail_scale=mixture_tail_scale,
    )
    yes_breakeven = float(candidate["yes_breakeven_probability"])
    no_upper_bound = float(candidate["no_required_yes_probability_upper_bound"])
    yes_edge = probability.model_p_yes - yes_breakeven
    no_edge = no_upper_bound - probability.model_p_yes
    display_forecast_mean = forecast_for_model.forecast_mean
    display_forecast_std = forecast_for_model.forecast_std
    if forecast_for_model.unit != classification.unit:
        display_forecast_mean = convert_temperature(
            forecast_for_model.forecast_mean,
            forecast_for_model.unit,
            classification.unit,
        )
        display_forecast_std = convert_temperature_std(
            forecast_for_model.forecast_std,
            forecast_for_model.unit,
            classification.unit,
        )

    validation = _validate_signal(
        classification=classification,
        forecast=forecast_for_model,
        as_of_time=parsed_as_of_time,
        forecast_issued_at=forecast_issued_at,
        generated_at=generated_at,
        resolution_analysis=resolution_analysis,
        as_of_time_defaulted=as_of_time_defaulted,
        extra_warnings=[forecast_time_warning] if forecast_time_warning else None,
        forecast_time_tolerance_seconds=forecast_time_tolerance_seconds,
    )
    warnings = _unique(
        list(classification.warnings)
        + list(probability.warnings)
        + list(validation.validation_warnings)
        + list(validation.data_provenance_warnings)
        + _resolution_warnings(resolution_analysis)
    )
    if yes_edge > edge_threshold and no_edge > edge_threshold:
        suggested_side: PaperSide = "NONE"
        warnings.append("market_data_conflict")
    elif yes_edge > edge_threshold:
        suggested_side = "YES"
    elif no_edge > edge_threshold:
        suggested_side = "NO"
    else:
        suggested_side = "NONE"
    if validation.signal_status not in {"VALID", "NEEDS_MANUAL_REVIEW"}:
        suggested_side = "NONE"

    signal = WeatherAlphaSignal(
        market_id=str(candidate.get("market_id") or ""),
        slug=candidate.get("slug"),
        question=str(candidate.get("question") or ""),
        category=candidate.get("category"),
        location_name=classification.location_name,
        metric=classification.metric,
        target_date=classification.target_date,
        threshold=classification.threshold_value,
        unit=classification.unit,
        bucket_mode=probability.bucket_mode,
        bucket_lower_bound=probability.bucket_lower_bound,
        bucket_upper_bound=probability.bucket_upper_bound,
        bucket_assumption=probability.bucket_assumption,
        forecast_mean=display_forecast_mean,
        forecast_std=display_forecast_std,
        model_p_yes=probability.model_p_yes,
        yes_breakeven=yes_breakeven,
        no_upper_bound=no_upper_bound,
        yes_model_edge=yes_edge,
        no_model_edge=no_edge,
        suggested_paper_side=suggested_side,
        edge_threshold=edge_threshold,
        confidence=_validated_confidence(classification.confidence, validation.signal_status),
        warnings=_unique(warnings),
        signal_status=validation.signal_status,
        validation_warnings=validation.validation_warnings,
        generated_at=generated_at,
        as_of_time=parsed_as_of_time,
        forecast_issued_at=forecast_issued_at,
        forecast_time_tolerance_seconds=forecast_time_tolerance_seconds,
        forecast_timing_tolerance_applied=validation.forecast_timing_tolerance_applied,
        resolution_source=validation.resolution_source,
        resolution_risk_score=validation.resolution_risk_score,
        ambiguity_risk=validation.ambiguity_risk,
        dispute_risk=validation.dispute_risk,
        source_location_name=validation.source_location_name,
        station_id=validation.forecast_station_id,
        forecast_station_id=validation.forecast_station_id,
        resolution_station_id=validation.resolution_station_id,
        bucket_boundary_confirmed=validation.bucket_numeric_boundary_confirmed,
        bucket_structure_confirmed=validation.bucket_structure_confirmed,
        bucket_numeric_boundary_confirmed=validation.bucket_numeric_boundary_confirmed,
        forecast_data_source=forecast_for_model.forecast_source or "unknown",
        forecast_source=forecast_for_model.forecast_source,
        forecast_model=forecast_for_model.forecast_model,
        std_method=forecast_for_model.std_method,
        latitude=forecast_for_model.latitude,
        longitude=forecast_for_model.longitude,
        actual_source=forecast_for_model.actual_source,
        cache_key=forecast_for_model.cache_key,
        raw_data_reference=forecast_for_model.raw_data_reference,
        data_provenance_warnings=validation.data_provenance_warnings,
        run_mode=run_mode,
        weather_model=probability.weather_model,
        model_parameters=probability.model_parameters,
        distribution_assumption=probability.distribution_assumption,
        calibration_group_key=calibration_result["calibration_group_key"],
        calibration_n=calibration_result["calibration_n"],
        calibration_bias=calibration_result["calibration_bias"],
        calibration_std_error=calibration_result["calibration_std_error"],
        calibration_applied=calibration_result["calibration_applied"],
        calibration_quality=calibration_result["calibration_quality"],
        calibration_quality_warnings=calibration_result["calibration_quality_warnings"],
        calibration_min_samples_required=calibration_result["calibration_min_samples_required"],
        calibration_bias_raw=calibration_result["calibration_bias_raw"],
        calibration_bias_shrunk=calibration_result["calibration_bias_shrunk"],
        calibration_std_error_raw=calibration_result["calibration_std_error_raw"],
        calibration_std_error_used=calibration_result["calibration_std_error_used"],
        forecast_mean_raw=calibration_result["forecast_mean_raw"],
        forecast_mean_adjusted=display_forecast_mean,
        forecast_std_raw=calibration_result["forecast_std_raw"],
        forecast_std_calibrated=display_forecast_std,
        strategy_score=_optional_float(candidate.get("strategy_score")),
    )
    signal.journal_draft_payload = _journal_draft_payload(
        signal=signal,
        candidate=candidate,
        side=suggested_side,
    )
    return signal


def write_weather_alpha_signals_json(
    result: WeatherAlphaScanResult,
    output_path: str | Path,
) -> int:
    """Write alpha signals to JSON and return signal count."""

    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    data = [signal.model_dump(mode="json") for signal in result.signals]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(result.signals)


def weather_alpha_report(result: WeatherAlphaScanResult) -> str:
    """Render weather alpha signals as Markdown."""

    lines = [
        "# Weather Temperature Alpha Signals",
        "",
        f"- Weather candidates identified: `{result.weather_candidate_count}`",
        f"- Threshold candidates identified: `{getattr(result, 'threshold_candidate_count', 0)}`",
        f"- Exact bucket candidates identified: `{getattr(result, 'exact_bucket_candidate_count', 0)}`",
        f"- Weather alpha signals generated: `{len(result.signals)}`",
        "- Scope: `temperature_threshold_and_exact_bucket_only`",
        "- Purpose: `paper_research_not_trade_signal`",
        "- Forecast actual_value is not used for model probability.",
        "",
    ]
    if result.skipped:
        lines.extend(["## Skipped Weather Candidates", ""])
        lines.extend(f"- `{item}`" for item in result.skipped[:20])
        lines.append("")
    for signal in result.signals:
        lines.extend(_signal_section(signal))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _signal_section(signal: WeatherAlphaSignal) -> list[str]:
    return [
        f"## {signal.question}",
        "",
        f"- market_id: `{signal.market_id}`",
        f"- slug: `{signal.slug or 'n/a'}`",
        f"- category: `{signal.category or 'n/a'}`",
        f"- location / metric / date: `{signal.location_name}` / `{signal.metric}` / `{signal.target_date}`",
        f"- threshold: `{signal.threshold:g}{signal.unit}`",
        f"- signal_status: `{signal.signal_status}`",
        f"- validation_warnings: `{_fmt_list(signal.validation_warnings)}`",
        f"- run_mode: `{signal.run_mode}`",
        f"- weather_model: `{signal.weather_model}`",
        f"- model_parameters: `{json.dumps(signal.model_parameters, sort_keys=True)}`",
        f"- distribution_assumption: `{signal.distribution_assumption}`",
        f"- calibration_group_key: `{signal.calibration_group_key or 'n/a'}`",
        f"- calibration_applied: `{signal.calibration_applied}`",
        f"- calibration_quality: `{signal.calibration_quality or 'n/a'}`",
        f"- calibration_quality_warnings: `{_fmt_list(signal.calibration_quality_warnings)}`",
        f"- calibration_n / min_required: `{signal.calibration_n if signal.calibration_n is not None else 'n/a'}` / `{signal.calibration_min_samples_required if signal.calibration_min_samples_required is not None else 'n/a'}`",
        f"- calibration bias raw/shrunk/used: `{_fmt_optional_float(signal.calibration_bias_raw)}` / `{_fmt_optional_float(signal.calibration_bias_shrunk)}` / `{_fmt_optional_float(signal.calibration_bias)}`",
        f"- calibration std raw/used: `{_fmt_optional_float(signal.calibration_std_error_raw)}` / `{_fmt_optional_float(signal.calibration_std_error_used)}`",
        f"- raw forecast_mean/std: `{_fmt_optional_float(signal.forecast_mean_raw)}` / `{_fmt_optional_float(signal.forecast_std_raw)}`",
        f"- adjusted forecast_mean/std: `{_fmt_optional_float(signal.forecast_mean_adjusted)}` / `{_fmt_optional_float(signal.forecast_std_calibrated)}`",
        f"- generated_at: `{signal.generated_at.isoformat()}`",
        f"- as_of_time: `{signal.as_of_time.isoformat() if signal.as_of_time else 'n/a'}`",
        f"- forecast_issued_at: `{signal.forecast_issued_at.isoformat() if signal.forecast_issued_at else 'n/a'}`",
        f"- forecast_time_tolerance_seconds: `{signal.forecast_time_tolerance_seconds:g}`",
        f"- forecast_timing_tolerance_applied: `{signal.forecast_timing_tolerance_applied}`",
        f"- forecast_source: `{signal.forecast_source or 'n/a'}`",
        f"- forecast_model: `{signal.forecast_model or 'n/a'}`",
        f"- std_method: `{signal.std_method or 'n/a'}`",
        f"- latitude / longitude: `{_fmt_optional_float(signal.latitude)}` / `{_fmt_optional_float(signal.longitude)}`",
        f"- actual_source: `{signal.actual_source or 'n/a'}`",
        f"- cache_key: `{signal.cache_key or 'n/a'}`",
        f"- raw_data_reference: `{signal.raw_data_reference or 'n/a'}`",
        f"- data provenance warnings: `{_fmt_list(signal.data_provenance_warnings)}`",
        f"- resolution_source: `{signal.resolution_source or 'n/a'}`",
        f"- resolution risk: `{signal.ambiguity_risk or 'n/a'}` ambiguity, `{signal.dispute_risk or 'n/a'}` dispute, score `{_fmt_optional_float(signal.resolution_risk_score)}`",
        f"- forecast_station_id: `{signal.forecast_station_id or 'n/a'}`",
        f"- resolution_station_id: `{signal.resolution_station_id or 'n/a'}`",
        f"- source location: `{signal.source_location_name or 'n/a'}`",
        f"- bucket_structure_confirmed: `{signal.bucket_structure_confirmed}`",
        f"- bucket_numeric_boundary_confirmed: `{signal.bucket_numeric_boundary_confirmed}`",
        f"- bucket_mode: `{signal.bucket_mode or 'n/a'}`",
        f"- bucket interval: `{_fmt_bucket_interval(signal)}`",
        f"- bucket assumption: `{signal.bucket_assumption or 'n/a'}`",
        f"- forecast mean/std: `{signal.forecast_mean:g}{signal.unit}` / `{signal.forecast_std:g}{signal.unit}`",
        f"- model_p_yes: `{signal.model_p_yes:.2%}`",
        f"- YES breakeven: `{signal.yes_breakeven:.2%}`",
        f"- NO profitable if p_yes < `{signal.no_upper_bound:.2%}`",
        f"- YES model edge: `{signal.yes_model_edge:.2%}`",
        f"- NO model edge: `{signal.no_model_edge:.2%}`",
        f"- suggested_paper_side: `{signal.suggested_paper_side}`",
        f"- warnings: `{_fmt_list(signal.warnings)}`",
        *(
            ["- Signal requires manual validation before journal entry."]
            if signal.signal_status != "VALID"
            else []
        ),
        *(
            ["- Forecast input is manual/sample data; do not treat as validated alpha."]
            if _is_manual_or_sample_source(signal.forecast_source)
            or _is_manual_std_method(signal.std_method)
            else []
        ),
        "",
        "### Journal Draft Payload",
        "",
        "```json",
        json.dumps(signal.journal_draft_payload, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
    ]


def _journal_draft_payload(
    *,
    signal: WeatherAlphaSignal,
    candidate: dict[str, Any],
    side: PaperSide,
) -> dict[str, Any]:
    invalid_research_only = signal.signal_status in {
        "INVALID_DATA",
        "RULE_MISMATCH",
        "STALE_OR_FUTURE_FORECAST",
        "LOCATION_SOURCE_MISMATCH",
    }
    if invalid_research_only:
        side = "NONE"
    if side == "NONE" and not invalid_research_only:
        return {}
    if side not in {"YES", "NO", "NONE"}:
        return {}
    draft_key = "journal_draft_payload_yes" if side == "YES" else "journal_draft_payload_no"
    source_draft = candidate.get(draft_key)
    if not isinstance(source_draft, dict):
        source_draft = {}
    entry_price = None if side == "NONE" else _optional_float(source_draft.get("entry_price"))
    fee_per_share = None if side == "NONE" else _optional_float(source_draft.get("fee_per_share"))
    entry_size = _optional_float(source_draft.get("entry_size")) or 0.0
    edge = None
    if side == "YES":
        edge = signal.yes_model_edge
    elif side == "NO":
        edge = signal.no_model_edge
    bucket_rationale = (
        f"; model assumes {signal.bucket_mode} integer bucket; "
        f"bucket_interval=[{signal.bucket_lower_bound:g}, {signal.bucket_upper_bound:g}){signal.unit}; "
        f"model_p_yes={signal.model_p_yes:.6g}"
        if signal.bucket_mode
        and signal.bucket_lower_bound is not None
        and signal.bucket_upper_bound is not None
        else f"; model_p_yes={signal.model_p_yes:.6g}"
    )
    validation_rationale = ""
    if signal.signal_status != "VALID":
        validation_rationale = (
            f"; signal_status={signal.signal_status}; "
            f"validation_warnings={_fmt_list(signal.validation_warnings)}; "
            f"data_provenance_warnings={_fmt_list(signal.data_provenance_warnings)}; "
            "requires manual validation before paper entry"
        )
    return _without_none(
        {
            "market_id": signal.market_id,
            "slug": signal.slug,
            "question": signal.question,
            "category": signal.category,
            "end_date": source_draft.get("end_date"),
            "candidate_score": source_draft.get("candidate_score") or candidate.get("candidate_score"),
            "candidate_grade": source_draft.get("candidate_grade") or candidate.get("candidate_grade"),
            "side": side,
            "fair_yes_probability": signal.model_p_yes,
            "probability_source": "weather_threshold_model",
            "entry_price": entry_price,
            "entry_size": entry_size,
            "fee_per_share": fee_per_share,
            "expected_value_per_share": edge,
            "expected_profit": edge * entry_size if edge is not None and entry_size else None,
            "rationale": (
                f"{WEATHER_ALPHA_RATIONALE}; forecast_mean={signal.forecast_mean:g}{signal.unit}; "
                f"forecast_std={signal.forecast_std:g}{signal.unit}; "
                f"threshold={signal.threshold:g}{signal.unit}; "
                f"weather_model={signal.weather_model}; "
                f"model_parameters={json.dumps(signal.model_parameters, sort_keys=True)}; "
                f"distribution_assumption={signal.distribution_assumption}; "
                f"calibration_applied={signal.calibration_applied}; "
                f"calibration_quality={signal.calibration_quality or 'n/a'}; "
                f"calibration_quality_warnings={_fmt_list(signal.calibration_quality_warnings)}; "
                f"calibration_group_key={signal.calibration_group_key or 'n/a'}; "
                f"calibration_n={signal.calibration_n if signal.calibration_n is not None else 'n/a'}; "
                f"calibration_min_samples_required={signal.calibration_min_samples_required if signal.calibration_min_samples_required is not None else 'n/a'}; "
                f"calibration_bias={signal.calibration_bias if signal.calibration_bias is not None else 'n/a'}; "
                f"calibration_bias_raw={signal.calibration_bias_raw if signal.calibration_bias_raw is not None else 'n/a'}; "
                f"calibration_bias_shrunk={signal.calibration_bias_shrunk if signal.calibration_bias_shrunk is not None else 'n/a'}; "
                f"calibration_std_error={signal.calibration_std_error if signal.calibration_std_error is not None else 'n/a'}; "
                f"calibration_std_error_raw={signal.calibration_std_error_raw if signal.calibration_std_error_raw is not None else 'n/a'}; "
                f"calibration_std_error_used={signal.calibration_std_error_used if signal.calibration_std_error_used is not None else 'n/a'}; "
                f"forecast_mean_raw={signal.forecast_mean_raw if signal.forecast_mean_raw is not None else 'n/a'}; "
                f"forecast_mean_adjusted={signal.forecast_mean_adjusted if signal.forecast_mean_adjusted is not None else 'n/a'}"
                f"{bucket_rationale}"
                f"{validation_rationale}"
            ),
            "status": "OPEN",
        }
    )


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _calibrated_forecast(
    *,
    forecast: WeatherForecast,
    classification: MarketClassification,
    calibration_summaries: CalibrationSummaries | None,
    calibration_group: list[str] | str,
    use_calibrated_std: bool,
    use_calibrated_bias: bool,
    min_calibration_samples: int,
    allow_low_quality_calibration: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "forecast": forecast,
        "calibration_group_key": None,
        "calibration_n": None,
        "calibration_bias": None,
        "calibration_std_error": None,
        "calibration_applied": False,
        "calibration_quality": None,
        "calibration_quality_warnings": [],
        "calibration_min_samples_required": None,
        "calibration_bias_raw": None,
        "calibration_bias_shrunk": None,
        "calibration_std_error_raw": None,
        "calibration_std_error_used": None,
        "forecast_mean_raw": forecast.forecast_mean,
        "forecast_std_raw": forecast.forecast_std,
    }
    if calibration_summaries is None:
        return result
    group_fields = normalize_group_by(calibration_group)
    group_key, key_warning = current_forecast_calibration_key(
        metric=classification.metric or forecast.metric,
        target_date=classification.target_date or forecast.date,
        forecast_issued_at=forecast.forecast_issued_at,
        group_by=group_fields,
        location=classification.location_name or forecast.location,
        station_id=forecast.station_id,
        forecast_source=forecast.forecast_source,
        forecast_model=forecast.forecast_model,
    )
    result["calibration_group_key"] = group_key
    warnings = list(forecast.provider_warnings)
    if key_warning:
        warnings.append(key_warning)
        result["forecast"] = forecast.model_copy(update={"provider_warnings": _unique(warnings)})
        return result
    summary = calibration_summaries.get(group_key or "")
    if summary is None:
        warnings.append("missing_calibration_group")
        result["forecast"] = forecast.model_copy(update={"provider_warnings": _unique(warnings)})
        return result
    runtime_quality = calibration_quality_for_n(summary.n, min_calibration_samples)
    quality_warnings = calibration_quality_warnings(runtime_quality)
    if runtime_quality == "LOW" and not allow_low_quality_calibration:
        quality_warnings = _unique(quality_warnings + ["low_quality_calibration"])
    bias_raw = summary.bias_raw if summary.bias_raw is not None else summary.mean_error
    bias_shrunk = summary.bias_shrunk if summary.bias_shrunk is not None else summary.bias
    std_error_raw = summary.std_error_raw if summary.std_error_raw is not None else summary.std_error
    std_error_used = summary.std_error_used if summary.std_error_used is not None else summary.std_error
    result.update(
        {
            "calibration_n": summary.n,
            "calibration_bias": bias_shrunk,
            "calibration_std_error": std_error_used,
            "calibration_quality": runtime_quality,
            "calibration_quality_warnings": quality_warnings,
            "calibration_min_samples_required": min_calibration_samples,
            "calibration_bias_raw": bias_raw,
            "calibration_bias_shrunk": bias_shrunk,
            "calibration_std_error_raw": std_error_raw,
            "calibration_std_error_used": std_error_used,
        }
    )
    if quality_warnings:
        warnings.extend(quality_warnings)
    if runtime_quality == "INSUFFICIENT":
        result["forecast"] = forecast.model_copy(update={"provider_warnings": _unique(warnings)})
        return result
    updates: dict[str, Any] = {}
    applied_tags: list[str] = []
    if use_calibrated_bias:
        updates["forecast_mean"] = forecast.forecast_mean + bias_shrunk
        applied_tags.append("bias")
    if use_calibrated_std:
        if std_error_used > 0:
            updates["forecast_std"] = std_error_used
            updates["std_method"] = "calibrated_historical_error"
            applied_tags.append("std")
        else:
            warnings.append("calibration_std_error_not_positive")
    if applied_tags:
        model = forecast.forecast_model or "unknown_model"
        updates["forecast_model"] = f"{model}+calibration_{'_'.join(applied_tags)}"
        result["calibration_applied"] = True
    if warnings:
        updates["provider_warnings"] = _unique(warnings)
    result["forecast"] = forecast.model_copy(update=updates) if updates else forecast
    return result


def _validate_signal(
    *,
    classification: MarketClassification,
    forecast: WeatherForecast,
    as_of_time: datetime,
    forecast_issued_at: datetime | None,
    generated_at: datetime,
    resolution_analysis: ResolutionAnalysis | None,
    as_of_time_defaulted: bool,
    extra_warnings: list[str] | None = None,
    forecast_time_tolerance_seconds: float = 0.0,
) -> WeatherSignalValidation:
    warnings: list[str] = []
    status: SignalStatus = "VALID"
    tolerance_applied = False
    tolerance_seconds = max(0.0, float(forecast_time_tolerance_seconds))
    if as_of_time_defaulted:
        warnings.append("as_of_time_defaulted_to_generated_at")
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    elif as_of_time > generated_at:
        warnings.append("as_of_time_after_generated_at_replay_mode")
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    if forecast.provider_warnings:
        warnings.extend(forecast.provider_warnings)
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    if not forecast.location_mapping_found:
        warnings.append("missing_location_mapping")
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    if extra_warnings:
        warnings.extend(extra_warnings)
        status = _status_at_least(status, "INVALID_DATA")
    if forecast_issued_at is None:
        warnings.append("missing_forecast_issued_at")
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    elif forecast_issued_at > as_of_time:
        delta_seconds = (forecast_issued_at - as_of_time).total_seconds()
        if tolerance_seconds > 0 and delta_seconds <= tolerance_seconds:
            warnings.append("forecast_issued_within_live_capture_tolerance")
            tolerance_applied = True
        else:
            warnings.append("forecast_issued_after_as_of_time")
            status = "STALE_OR_FUTURE_FORECAST"

    target_date = _parse_date(classification.target_date)
    if target_date is not None and target_date < as_of_time.date():
        warnings.append("target_date_before_as_of_date")
        status = _status_at_least(status, "INVALID_DATA")

    source_text = _resolution_text_for_validation(resolution_analysis)
    resolution_source = resolution_analysis.resolution_source if resolution_analysis else None
    provenance_warnings = _data_provenance_warnings(forecast, resolution_source or source_text)
    if provenance_warnings:
        status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
    resolution_station_id = _extract_station_id(source_text)
    forecast_station_id = (forecast.station_id or "").strip().upper() or None
    source_location_name = _extract_source_location_name(source_text)
    bucket_structure_confirmed = False
    bucket_numeric_boundary_confirmed = False
    if classification.market_type == MarketType.weather_temperature_exact_bucket:
        bucket_structure_confirmed = _has_bucket_range_language(source_text)
        bucket_numeric_boundary_confirmed = _has_explicit_bucket_boundary(
            source_text,
            classification.threshold_value,
        )
        if bucket_structure_confirmed:
            if not bucket_numeric_boundary_confirmed:
                warnings.append("bucket_boundary_inferred_not_explicit")
                status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
        else:
            warnings.append("bucket_boundary_assumption_unconfirmed")
            status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")

    if resolution_analysis is not None:
        if resolution_analysis.ambiguity_risk in {"MEDIUM", "HIGH"} or resolution_analysis.dispute_risk in {
            "MEDIUM",
            "HIGH",
        }:
            warnings.append("resolution_risk_not_low")
            status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
        if resolution_analysis.warnings:
            warnings.extend(f"resolution_warning:{warning}" for warning in resolution_analysis.warnings)
    if _source_mentions_weather_station(source_text):
        if resolution_station_id and forecast_station_id is None:
            warnings.append("missing_forecast_station_id_for_resolution_station")
            warnings.append("missing_station_id_for_resolution_source")
            if "missing_forecast_station_id_for_resolution_station" not in provenance_warnings:
                provenance_warnings.append("missing_forecast_station_id_for_resolution_station")
            if "missing_station_id_for_station_resolution" not in provenance_warnings:
                provenance_warnings.append("missing_station_id_for_station_resolution")
            if source_location_name and classification.location_name != source_location_name:
                warnings.append("location_source_requires_manual_mapping")
            status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")
        elif resolution_station_id and forecast_station_id != resolution_station_id:
            warnings.append("station_id_mismatch")
            status = "LOCATION_SOURCE_MISMATCH"
        elif (
            source_location_name
            and classification.location_name != source_location_name
            and not forecast_station_id
        ):
            warnings.append("location_source_requires_manual_mapping")
            status = _status_at_least(status, "NEEDS_MANUAL_REVIEW")

    return WeatherSignalValidation(
        signal_status=status,
        validation_warnings=_unique(warnings + provenance_warnings),
        resolution_source=resolution_source,
        resolution_risk_score=resolution_analysis.risk_score if resolution_analysis else None,
        ambiguity_risk=resolution_analysis.ambiguity_risk if resolution_analysis else None,
        dispute_risk=resolution_analysis.dispute_risk if resolution_analysis else None,
        source_location_name=forecast.source_location_name or source_location_name,
        station_id=forecast_station_id,
        forecast_station_id=forecast_station_id,
        resolution_station_id=resolution_station_id,
        bucket_boundary_confirmed=bucket_numeric_boundary_confirmed,
        bucket_structure_confirmed=bucket_structure_confirmed,
        bucket_numeric_boundary_confirmed=bucket_numeric_boundary_confirmed,
        data_provenance_warnings=_unique(provenance_warnings),
        forecast_timing_tolerance_applied=tolerance_applied,
    )


def _resolution_warnings(resolution_analysis: ResolutionAnalysis | None) -> list[str]:
    if resolution_analysis is None:
        return []
    return [f"resolution_warning:{warning}" for warning in resolution_analysis.warnings]


def _data_provenance_warnings(forecast: WeatherForecast, resolution_source: str | None) -> list[str]:
    warnings: list[str] = []
    forecast_source = _normalized_optional(forecast.forecast_source)
    std_method = _normalized_optional(forecast.std_method)
    if forecast_source is None:
        warnings.append("missing_forecast_source")
    elif forecast_source in {"manual", "sample"}:
        warnings.append("manual_or_sample_forecast_source")
    if std_method is None:
        warnings.append("missing_std_method")
    elif std_method == "manual_assumption":
        warnings.append("manual_std_assumption")
    elif std_method == "configured_std":
        warnings.append("configured_std_used")
    elif std_method == "fallback_error_std":
        warnings.append("fallback_std_used")
    if forecast.latitude is None or forecast.longitude is None:
        warnings.append("missing_coordinates")
    if forecast.actual_value is not None:
        warnings.append("actual_value_present_not_used")
    if forecast.actual_value is not None and _normalized_optional(forecast.actual_source) is None:
        warnings.append("missing_actual_source")
    if _source_mentions_weather_station(resolution_source or ""):
        if not forecast.station_id:
            warnings.append("missing_forecast_station_id_for_resolution_station")
            warnings.append("missing_station_id_for_station_resolution")
        elif _source_mentions_wunderground(resolution_source) and forecast_source not in {
            "wunderground",
            "manual",
            "sample",
        }:
            warnings.append("forecast_source_mismatch")
    return _unique(warnings)


def _normalized_optional(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip().casefold()


def _is_manual_or_sample_source(value: str | None) -> bool:
    return _normalized_optional(value) in {"manual", "sample"}


def _is_manual_std_method(value: str | None) -> bool:
    return _normalized_optional(value) == "manual_assumption"


def _validated_confidence(confidence: float, status: SignalStatus) -> float:
    if status == "VALID":
        return confidence
    if status == "NEEDS_MANUAL_REVIEW":
        return min(confidence, 0.6)
    return min(confidence, 0.2)


def _status_at_least(current: SignalStatus, new_status: SignalStatus) -> SignalStatus:
    priority = {
        "VALID": 0,
        "NEEDS_MANUAL_REVIEW": 1,
        "MISSING_FORECAST_DATA": 2,
        "RULE_MISMATCH": 3,
        "LOCATION_SOURCE_MISMATCH": 4,
        "STALE_OR_FUTURE_FORECAST": 5,
        "INVALID_DATA": 6,
    }
    return new_status if priority[new_status] > priority[current] else current


def _source_mentions_weather_station(text: str) -> bool:
    lower = text.casefold()
    return any(term in lower for term in ("wunderground", "station", "airport", "limc"))


def _source_mentions_wunderground(text: str | None) -> bool:
    return "wunderground" in (text or "").casefold()


def _extract_station_id(text: str) -> str | None:
    if not text:
        return None
    for match in re.finditer(r"\b[A-Z]{4}\b", text):
        value = match.group(0)
        if value not in {"THIS", "WILL"}:
            return value
    return None


def _extract_source_location_name(text: str) -> str | None:
    if not text:
        return None
    match = re.search(
        r"((?:[A-Z][A-Za-z]+(?:\s+|$)){1,5}(?:Intl|International)?\s*Airport\s+Station)",
        text,
    )
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Station)", text)
    if match:
        return " ".join(match.group(1).split())
    return None


def _has_bucket_range_language(text: str) -> bool:
    lower = text.casefold()
    return any(term in lower for term in ("temperature range", "range that contains", "contains the highest temperature"))


def _has_explicit_bucket_boundary(text: str, threshold: float | None) -> bool:
    if threshold is None:
        return False
    lower = text.casefold()
    lower_bound = threshold - 0.5
    upper_bound = threshold + 0.5
    patterns = [
        f"{lower_bound:g} to {upper_bound:g}",
        f"{lower_bound:g}-{upper_bound:g}",
        f"[{lower_bound:g}, {upper_bound:g})",
        f"{lower_bound:g} <= ",
    ]
    return any(pattern in lower for pattern in patterns)


def _resolution_text_for_validation(resolution_analysis: ResolutionAnalysis | None) -> str:
    if resolution_analysis is None:
        return ""
    parts = [
        resolution_analysis.resolution_text_excerpt,
        resolution_analysis.resolution_source,
        resolution_analysis.deadline,
        resolution_analysis.what_counts_as_yes,
        resolution_analysis.what_counts_as_no,
    ]
    return " ".join(part for part in parts if part)


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return _aware_datetime(value)
    return _aware_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _parse_optional_datetime(value: str | None) -> tuple[datetime | None, str | None]:
    if value is None or not str(value).strip():
        return None, None
    try:
        return _parse_datetime(value), None
    except ValueError:
        return None, "invalid_forecast_issued_at"


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _fmt_bucket_interval(signal: WeatherAlphaSignal) -> str:
    if signal.bucket_lower_bound is None or signal.bucket_upper_bound is None:
        return "n/a"
    return f"[{signal.bucket_lower_bound:g}, {signal.bucket_upper_bound:g}){signal.unit}"


def _fmt_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
