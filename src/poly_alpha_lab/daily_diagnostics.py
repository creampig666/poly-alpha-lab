"""Read-only diagnostics for weather daily capture outputs."""

from __future__ import annotations

import json
import re
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from poly_alpha_lab.market_type_classifier import MarketType, classify_strategy_candidate
from poly_alpha_lab.resolution_analyzer import extract_weather_station_details
from poly_alpha_lab.weather_data import LocationResolver

SUPPORTED_WEATHER_TYPES = {
    MarketType.weather_temperature_threshold,
    MarketType.weather_temperature_exact_bucket,
    MarketType.weather_temperature_range_bucket,
}

EDGE_BUCKETS = (
    ("<0", float("-inf"), 0.0),
    ("0-1%", 0.0, 0.01),
    ("1-3%", 0.01, 0.03),
    ("3-5%", 0.03, 0.05),
    (">5%", 0.05, float("inf")),
)

REASON_ZH = {
    "not_weather_candidate": "不是天气候选市场。",
    "unsupported_weather_market_type": "看起来像天气市场，但当前 alpha 模块暂不支持该题型。",
    "missing_location": "无法从问题文本解析地点。",
    "missing_target_date": "无法从问题文本解析目标日期。",
    "missing_forecast": "没有找到匹配的 forecast 数据，或 provider 未返回可用 forecast。",
    "missing_orderbook": "缺少可执行盘口价格。",
    "insufficient_depth": "盘口深度不足，无法按指定 size 完整成交。",
    "no_model_probability": "没有生成模型概率。",
    "edge_below_threshold": "模型 edge 未超过当前阈值。",
    "validation_not_valid": "信号状态不是 VALID，存在数据、规则或来源风险。",
    "no_paper_side": "模型没有给出 YES/NO 方向，通常是 edge 没超过阈值。",
    "station_not_matched": "forecast 站点与 Polymarket 结算站点不一致或未能确认匹配。",
    "bucket_boundary_not_confirmed": "温度桶边界没有被规则明确确认。",
    "resolution_risk_not_low": "resolution risk 不是 LOW，需要人工复核。",
    "forecast_timing_issue": "forecast 时间与 as_of_time 存在时序风险或容忍窗口提示。",
    "unknown_reason": "无法从现有 JSON 字段推断原因。",
}

SKIPPED_REASON_ZH = {
    "no_paper_side": "模型没有给出 YES/NO 方向，通常是 edge 没超过阈值。",
    "signal_status_not_valid": "信号状态不是 VALID，存在数据、规则或来源风险。",
    "forecast_issued_within_live_capture_tolerance": (
        "live capture 中 forecast 时间略晚于 as_of_time，但在容忍窗口内，不视为未来函数。"
    ),
    "forecast_after_as_of_time": "forecast 时间晚于 as_of_time，存在未来数据风险。",
    "forecast_issued_after_as_of_time": "forecast 时间晚于 as_of_time，存在未来数据风险。",
    "station_not_matched": "模型 forecast 站点与 Polymarket 结算站点不一致。",
    "bucket_boundary_not_confirmed": "市场是温度桶，但具体数值边界没有被规则明确确认。",
    "sample_or_manual_forecast": "forecast 数据是 sample/manual，不可作为严格数据源。",
    "manual_std_method": "forecast_std 是手动假设，不可作为严格校准。",
    "calibration_quality_insufficient": "历史校准样本质量不足。",
    "calibration_samples_too_low": "校准样本数太少。",
}


def diagnose_weather_daily_captures(
    daily_dir: str | Path = "data/daily",
    *,
    days: int = 7,
    language: str = "zh",
) -> dict[str, Any]:
    """Aggregate recent daily weather capture artifacts into diagnostics."""

    del language
    root = Path(daily_dir)
    selected_dirs = _recent_day_dirs(root, days)
    runs: list[dict[str, Any]] = []
    totals = {
        "strategy_candidates": 0,
        "weather_candidates": 0,
        "weather_alpha_signals": 0,
        "paper_side_yes_no": 0,
        "strict_saved": 0,
    }
    side_counts: Counter[str] = Counter()
    signal_status_counts: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    inferred_reasons: Counter[str] = Counter()
    market_type_counts: Counter[str] = Counter()
    market_type_examples: dict[str, str] = {}
    location_counts: Counter[str] = Counter()
    forecast_source_counts: Counter[str] = Counter()
    std_method_counts: Counter[str] = Counter()
    calibration_quality_counts: Counter[str] = Counter()
    source_location_counts: Counter[str] = Counter()
    edge_bucket_counts: Counter[str] = Counter()
    edge_records: list[dict[str, Any]] = []
    max_yes_edge: float | None = None
    max_no_edge: float | None = None
    forecast_station_missing = 0
    resolution_station_missing = 0
    station_mismatch = 0
    missing_forecast_count = 0
    missing_forecast_location_counts: Counter[str] = Counter()
    calibration_applied_count = 0
    timing_issue_counts: Counter[str] = Counter()

    for summary_path in _summary_files(selected_dirs):
        summary = _load_json_object(summary_path)
        if summary is None:
            continue
        run_dir = summary_path.parent
        strategy_path = _resolve_artifact_path(summary.get("strategy_candidates_path"), run_dir)
        alpha_path = _resolve_artifact_path(summary.get("weather_alpha_path"), run_dir)
        strategy_candidates = _load_json_list(strategy_path) if strategy_path else []
        signals = _load_json_list(alpha_path) if alpha_path else []
        signal_by_market_id = {str(signal.get("market_id")): signal for signal in signals}

        run = _run_row(summary, summary_path, strategy_candidates, signals)
        runs.append(run)

        totals["strategy_candidates"] += int(summary.get("strategy_candidates_count") or len(strategy_candidates))
        totals["weather_candidates"] += int(summary.get("weather_candidates_count") or 0)
        totals["weather_alpha_signals"] += int(summary.get("weather_alpha_signals_count") or len(signals))
        totals["strict_saved"] += int(summary.get("backtest_saved") or 0)
        skipped_reasons.update(_counter_from_mapping(summary.get("skipped_reasons")))

        for signal in signals:
            side = str(signal.get("suggested_paper_side") or "UNKNOWN")
            side_counts[side] += 1
            if side in {"YES", "NO"}:
                totals["paper_side_yes_no"] += 1
            signal_status_counts[str(signal.get("signal_status") or "UNKNOWN")] += 1
            _record_signal_diagnostics(
                signal,
                edge_bucket_counts=edge_bucket_counts,
                edge_records=edge_records,
                forecast_source_counts=forecast_source_counts,
                std_method_counts=std_method_counts,
                calibration_quality_counts=calibration_quality_counts,
                source_location_counts=source_location_counts,
                timing_issue_counts=timing_issue_counts,
            )
            if signal.get("calibration_applied"):
                calibration_applied_count += 1
            yes_edge = _float_or_none(signal.get("yes_model_edge"))
            no_edge = _float_or_none(signal.get("no_model_edge"))
            if yes_edge is not None:
                max_yes_edge = yes_edge if max_yes_edge is None else max(max_yes_edge, yes_edge)
            if no_edge is not None:
                max_no_edge = no_edge if max_no_edge is None else max(max_no_edge, no_edge)
            forecast_station = _clean_string(signal.get("forecast_station_id") or signal.get("station_id"))
            resolution_station = _clean_string(signal.get("resolution_station_id"))
            if not forecast_station:
                forecast_station_missing += 1
            if not resolution_station:
                resolution_station_missing += 1
            if forecast_station and resolution_station and forecast_station != resolution_station:
                station_mismatch += 1
            if _has_any_warning(signal, {"station_not_matched", "missing_forecast_station_id_for_resolution_station"}):
                station_mismatch += 1

        for candidate in strategy_candidates:
            if not isinstance(candidate, dict):
                continue
            classification = classify_strategy_candidate(candidate)
            weather_relevant = (
                classification.market_type in SUPPORTED_WEATHER_TYPES
                or _weather_looking_candidate(candidate)
            )
            if weather_relevant:
                market_type = _market_type_bucket(candidate, classification)
                market_type_counts[market_type] += 1
                market_type_examples.setdefault(market_type, str(candidate.get("question") or "n/a"))
            if classification.location_name:
                location_counts[str(classification.location_name)] += 1
            reasons, inferred = _candidate_failure_reasons(candidate, classification, signal_by_market_id)
            failure_reasons.update(reasons)
            inferred_reasons.update(inferred)
            if "missing_forecast" in reasons:
                missing_forecast_count += 1
                missing_forecast_location_counts[classification.location_name or "unknown"] += 1

    funnel = _funnel_rows(totals)
    top_bottlenecks = _top_bottlenecks(
        failure_reasons,
        skipped_reasons,
        totals=totals,
        station_mismatch=station_mismatch,
    )
    result = {
        "daily_dir": str(root),
        "days": days,
        "selected_dates": [path.name for path in selected_dirs],
        "runs": sorted(runs, key=lambda item: item.get("captured_at") or ""),
        "funnel": funnel,
        "totals": totals,
        "suggested_paper_side_counts": dict(sorted(side_counts.items())),
        "signal_status_counts": dict(sorted(signal_status_counts.items())),
        "backtest_skipped_reasons": dict(skipped_reasons.most_common()),
        "failure_reason_counts": dict(failure_reasons.most_common()),
        "inferred_reason_counts": dict(inferred_reasons.most_common()),
        "edge_distribution": {
            "max_yes_model_edge": max_yes_edge,
            "max_no_model_edge": max_no_edge,
            "bucket_counts": {label: edge_bucket_counts.get(label, 0) for label, *_ in EDGE_BUCKETS},
            "records": edge_records,
        },
        "market_type_counts": dict(market_type_counts.most_common()),
        "market_type_examples": market_type_examples,
        "location_counts": dict(location_counts.most_common()),
        "station_diagnostics": {
            "forecast_station_id_missing": forecast_station_missing,
            "resolution_station_id_missing": resolution_station_missing,
            "station_mismatch": station_mismatch,
            "source_location_name_counts": dict(source_location_counts.most_common()),
        },
        "forecast_diagnostics": {
            "forecast_source_counts": dict(forecast_source_counts.most_common()),
            "std_method_counts": dict(std_method_counts.most_common()),
            "calibration_applied_count": calibration_applied_count,
            "calibration_quality_counts": dict(calibration_quality_counts.most_common()),
            "missing_forecast_count": missing_forecast_count,
            "timing_issue_counts": dict(timing_issue_counts.most_common()),
            "top_missing_locations_or_stations": dict(missing_forecast_location_counts.most_common(10)),
            "locations_pending_path": "data/weather/locations_pending.csv",
            "locations_pending_count": _count_csv_rows(Path("data/weather/locations_pending.csv")),
        },
        "top_bottlenecks": top_bottlenecks,
        "recommendations": _recommendations(totals, failure_reasons, station_mismatch),
    }
    return result


def write_weather_diagnostics_json(result: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_weather_diagnostics_markdown(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    language: str = "zh",
) -> None:
    if language != "zh":
        raise ValueError("Only zh diagnostics report is supported in v1.3.4")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(weather_diagnostics_report_zh(result), encoding="utf-8")


def diagnose_missing_forecasts(
    daily_dir: str | Path = "data/daily",
    *,
    days: int = 7,
    locations_file: str | Path = "data/weather/locations.csv",
    pending_output: str | Path = "data/weather/locations_pending.csv",
    language: str = "zh",
) -> dict[str, Any]:
    """Build a candidate-level report for weather markets that lacked forecasts."""

    del language
    root = Path(daily_dir)
    selected_dirs = _recent_day_dirs(root, days)
    resolver = LocationResolver(locations_file)
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    location_station_counts: Counter[str] = Counter()
    examples: dict[str, str] = {}

    for summary_path in _summary_files(selected_dirs):
        summary = _load_json_object(summary_path)
        if summary is None:
            continue
        run_dir = summary_path.parent
        captured_at = str(summary.get("captured_at") or "")
        strategy_path = _resolve_artifact_path(summary.get("strategy_candidates_path"), run_dir)
        alpha_path = _resolve_artifact_path(summary.get("weather_alpha_path"), run_dir)
        strategy_candidates = _load_json_list(strategy_path) if strategy_path else []
        signals = _load_json_list(alpha_path) if alpha_path else []
        signal_ids = {str(signal.get("market_id")) for signal in signals}
        skipped_keys = _missing_forecast_lookup_keys(signals, summary)

        for candidate in strategy_candidates:
            if not isinstance(candidate, dict):
                continue
            classification = classify_strategy_candidate(candidate)
            if classification.market_type not in SUPPORTED_WEATHER_TYPES and not _weather_looking_candidate(candidate):
                continue
            parsed_location = classification.location_name or _rough_location_from_question(
                str(candidate.get("question") or "")
            )
            market_id = str(candidate.get("market_id") or candidate.get("id") or "")
            if market_id in signal_ids:
                continue
            if classification.market_type not in SUPPORTED_WEATHER_TYPES:
                reason = (
                    "location_not_in_locations_csv"
                    if parsed_location and resolver.resolve(parsed_location) is None
                    else ("parse_error" if classification.warnings else "metric_not_supported")
                )
            else:
                reason = _missing_forecast_reason(
                    candidate=candidate,
                    classification=classification,
                    resolver=resolver,
                    captured_at=captured_at,
                )
            station = _station_details_for_candidate(candidate)
            lookup_station = station.station_id
            lookup_key = (
                f"{lookup_station or parsed_location or 'unknown'}:"
                f"{classification.target_date or 'unknown'}:{classification.metric or 'unknown'}"
            )
            if skipped_keys and lookup_key not in skipped_keys and classification.market_type in SUPPORTED_WEATHER_TYPES:
                # Existing alpha skipped strings are not guaranteed to include every field, so this remains diagnostic only.
                pass
            match = resolver.resolve(
                parsed_location or "",
                station_id=lookup_station,
            )
            record = {
                "date": run_dir.name,
                "captured_at": captured_at,
                "market_id": market_id,
                "slug": candidate.get("slug"),
                "question": candidate.get("question"),
                "category": candidate.get("category"),
                "parsed_market_type": classification.market_type.value,
                "parsed_location_name": parsed_location,
                "metric": classification.metric,
                "target_date": classification.target_date,
                "threshold": classification.threshold_value,
                "range_lower": classification.range_lower,
                "range_upper": classification.range_upper,
                "unit": classification.unit,
                "resolution_source": _candidate_resolution_source(candidate),
                "extracted_station_id": lookup_station,
                "extracted_station_name": station.station_name,
                "forecast_lookup_key": lookup_key,
                "locations_csv_match": "yes" if match else "no",
                "reason": reason,
            }
            records.append(record)
            reason_counts[reason] += 1
            group_key = lookup_station or parsed_location or "unknown"
            location_station_counts[group_key] += 1
            examples.setdefault(group_key, str(candidate.get("question") or "n/a"))

    pending_rows = _write_locations_pending(records, pending_output)
    suggestions_summary = _locations_suggestions_summary(Path("data/weather/locations_suggestions.csv"))
    return {
        "daily_dir": str(root),
        "days": days,
        "selected_dates": [path.name for path in selected_dirs],
        "records": records,
        "reason_counts": dict(reason_counts.most_common()),
        "location_station_counts": dict(location_station_counts.most_common()),
        "examples": examples,
        "locations_file": str(locations_file),
        "locations_pending_path": str(pending_output),
        "locations_pending_new_or_updated": pending_rows,
        "locations_suggestions": suggestions_summary,
        "top_missing_locations_or_stations": [
            {
                "location_or_station": key,
                "count": count,
                "example_question": examples.get(key, "n/a"),
                "suggested_action": _missing_forecast_suggested_action(key, records),
            }
            for key, count in location_station_counts.most_common(10)
        ],
    }


def write_missing_forecasts_csv(result: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "captured_at",
        "market_id",
        "slug",
        "question",
        "category",
        "parsed_market_type",
        "parsed_location_name",
        "metric",
        "target_date",
        "threshold",
        "range_lower",
        "range_upper",
        "unit",
        "resolution_source",
        "extracted_station_id",
        "extracted_station_name",
        "forecast_lookup_key",
        "locations_csv_match",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in result.get("records", []):
            writer.writerow({field: record.get(field) for field in fields})


def write_missing_forecasts_markdown(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    language: str = "zh",
) -> None:
    if language != "zh":
        raise ValueError("Only zh missing forecast report is supported in v1.3.5")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(missing_forecasts_report_zh(result), encoding="utf-8")


def missing_forecasts_report_zh(result: dict[str, Any]) -> str:
    lines = [
        "# 天气 Forecast 缺失明细诊断",
        "",
        "本报告只诊断 forecast/location/station 覆盖率，不改变交易逻辑、edge threshold 或 strict gate。",
        "",
        "## 1. 总览",
        "",
        f"- 最近天数：`{result.get('days')}`",
        f"- missing forecast 明细数：`{len(result.get('records', []))}`",
        f"- locations_pending.csv：`{result.get('locations_pending_path')}`",
        f"- 待人工补坐标条目数：`{result.get('locations_pending_new_or_updated')}`",
        f"- locations_suggestions.csv：`{result.get('locations_suggestions', {}).get('path')}`",
        f"- 可 promote 的 high-confidence 建议数：`{result.get('locations_suggestions', {}).get('high_confidence_promotable', 0)}`",
        "",
        "## 2. 按原因统计",
        "",
        "| reason | 次数 | 中文解释 |",
        "|---|---:|---|",
    ]
    reason_counts = result.get("reason_counts", {})
    if reason_counts:
        for reason, count in reason_counts.items():
            lines.append(f"| `{reason}` | {count} | {_missing_reason_zh(reason)} |")
    else:
        lines.append("| 无 | 0 | 没有发现 missing_forecast 候选。 |")

    lines.extend(
        [
            "",
            "## 3. Top missing locations/stations",
            "",
            "| location/station | missing 次数 | 示例市场 | 建议动作 |",
            "|---|---:|---|---|",
        ]
    )
    top = result.get("top_missing_locations_or_stations", [])
    if top:
        for item in top:
            lines.append(
                "| {key} | {count} | {example} | {action} |".format(
                    key=item.get("location_or_station"),
                    count=item.get("count"),
                    example=str(item.get("example_question") or "").replace("|", "/"),
                    action=item.get("suggested_action"),
                )
            )
    else:
        lines.append("| 无 | 0 | 无 | 无需动作 |")

    lines.extend(
        [
            "",
            "## 4. 明细样例",
            "",
            "| date | market_id | location | station_id | metric | target_date | type | reason | question |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in result.get("records", [])[:30]:
        lines.append(
            "| {date} | {market_id} | {location} | {station} | {metric} | {target_date} | {market_type} | {reason} | {question} |".format(
                date=record.get("date") or "n/a",
                market_id=record.get("market_id") or "n/a",
                location=record.get("parsed_location_name") or "n/a",
                station=record.get("extracted_station_id") or "n/a",
                metric=record.get("metric") or "n/a",
                target_date=record.get("target_date") or "n/a",
                market_type=record.get("parsed_market_type") or "n/a",
                reason=record.get("reason") or "n/a",
                question=str(record.get("question") or "").replace("|", "/"),
            )
        )
    if not result.get("records"):
        lines.append("| 无 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | 无 |")

    lines.extend(
        [
            "",
            "## 5. 下一步建议",
            "",
        ]
    )
    reason_counts = Counter(result.get("reason_counts", {}))
    if reason_counts.get("station_not_in_locations_csv") or reason_counts.get("location_not_in_locations_csv"):
        lines.append("- 先人工补齐 `data/weather/locations_pending.csv` 中高频 station/location 的经纬度，再复制到正式 `locations.csv`。")
    suggestions = result.get("locations_suggestions", {})
    if suggestions.get("exists") and suggestions.get("high_confidence_promotable", 0):
        lines.append("- 已发现 high-confidence suggestions，可运行 `weather-locations promote-suggestions` 生成 `locations_updated.csv`；审查后再替换正式 `locations.csv`。")
    if reason_counts.get("parse_error") or reason_counts.get("metric_not_supported"):
        lines.append("- 对 unsupported wording 做样例归档，再决定是否继续扩展 classifier。")
    if reason_counts.get("target_date_out_of_provider_range"):
        lines.append("- 检查 Open-Meteo current forecast 的可用日期窗口，过远或已过日期不应期待 current provider 返回 forecast。")
    if not reason_counts:
        lines.append("- 当前 missing forecast 不是主要瓶颈，继续观察 edge 和 strict gate skip reasons。")
    lines.append("- 不要因为 coverage 诊断结果而放宽 strict gate 或降低 edge threshold。")
    return "\n".join(lines) + "\n"


def weather_diagnostics_report_zh(result: dict[str, Any]) -> str:
    totals = result.get("totals", {})
    lines = [
        "# Polymarket 天气 Daily Capture 诊断报告",
        "",
        "## 1. 运行稳定性",
        "",
        "| 日期 | captured_at | strategy | weather | alpha | 盘口快照 | saved | skipped | errors | warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in result.get("runs", []):
        lines.append(
            "| {date} | {captured_at} | {strategy} | {weather} | {alpha} | {snapshots} | "
            "{saved} | {skipped} | {errors} | {warnings} |".format(
                date=run.get("date", "n/a"),
                captured_at=run.get("captured_at", "n/a"),
                strategy=run.get("strategy_candidates_count", 0),
                weather=run.get("weather_candidates_count", 0),
                alpha=run.get("weather_alpha_signals_count", 0),
                snapshots=run.get("snapshots_inserted", 0),
                saved=run.get("backtest_saved", 0),
                skipped=run.get("backtest_skipped", 0),
                errors=_join_or_none(run.get("errors")),
                warnings=_join_or_none(run.get("warnings")),
            )
        )

    lines.extend(
        [
            "",
            "## 2. 漏斗分析",
            "",
            "| 阶段 | 数量 | 转化率 | 解释 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in result.get("funnel", []):
        lines.append(
            f"| {row['stage']} | {row['count']} | {row['conversion_rate']} | {row['explanation']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 为什么 alpha signal 少？",
            "",
            "### 主要瓶颈",
            "",
        ]
    )
    bottlenecks = result.get("top_bottlenecks", [])
    if bottlenecks:
        lines.extend(["| 瓶颈 | 次数 | 中文解释 |", "|---|---:|---|"])
        for item in bottlenecks:
            reason = item.get("reason", "unknown_reason")
            lines.append(f"| `{reason}` | {item.get('count', 0)} | {_reason_zh(reason)} |")
    else:
        lines.append("暂无明显瓶颈。")

    lines.extend(["", "### 失败层级统计", "", "| 失败层级 | 次数 | 中文解释 |", "|---|---:|---|"])
    failure_counts = result.get("failure_reason_counts", {})
    if failure_counts:
        for reason, count in failure_counts.items():
            inferred = result.get("inferred_reason_counts", {}).get(reason, 0)
            suffix = f"；其中 {inferred} 条为 inferred" if inferred else ""
            lines.append(f"| `{reason}` | {count} | {_reason_zh(reason)}{suffix} |")
    else:
        lines.append("| 无 | 0 | 无 |")

    lines.extend(
        [
            "",
            "## 4. Edge 分布",
            "",
        ]
    )
    edge = result.get("edge_distribution", {})
    if int(totals.get("weather_alpha_signals") or 0) == 0:
        lines.append("没有 alpha signal，无法统计 edge；需要先诊断 candidate 到 alpha 的失败原因。")
    else:
        lines.extend(
            [
                f"- 最大 yes_model_edge：`{_fmt_float(edge.get('max_yes_model_edge'))}`",
                f"- 最大 no_model_edge：`{_fmt_float(edge.get('max_no_model_edge'))}`",
                "",
                "| edge 区间 | signal 数 |",
                "|---|---:|",
            ]
        )
        for label, count in edge.get("bucket_counts", {}).items():
            lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## 5. 市场类型统计",
            "",
            "| 类型 | 数量 | 示例问题 |",
            "|---|---:|---|",
        ]
    )
    for market_type, count in result.get("market_type_counts", {}).items():
        example = result.get("market_type_examples", {}).get(market_type, "n/a")
        lines.append(f"| {market_type} | {count} | {example} |")
    if not result.get("market_type_counts"):
        lines.append("| 无 | 0 | 无 |")

    station = result.get("station_diagnostics", {})
    lines.extend(
        [
            "",
            "## 6. 地点 / 站点匹配",
            "",
            f"- forecast_station_id 缺失数量：`{station.get('forecast_station_id_missing', 0)}`",
            f"- resolution_station_id 缺失数量：`{station.get('resolution_station_id_missing', 0)}`",
            f"- station mismatch 数量：`{station.get('station_mismatch', 0)}`",
            "",
            "| location_name | 数量 |",
            "|---|---:|",
        ]
    )
    for location, count in result.get("location_counts", {}).items():
        lines.append(f"| {location} | {count} |")
    if not result.get("location_counts"):
        lines.append("| 无 | 0 |")
    if int(station.get("station_mismatch") or 0) > 0:
        lines.append("")
        lines.append("当前主要瓶颈可能包含 forecast location 与 Polymarket resolution station 不匹配，需要扩展 locations.csv station mapping。")

    forecast = result.get("forecast_diagnostics", {})
    lines.extend(
        [
            "",
            "## 7. Forecast 数据问题",
            "",
            f"- calibration_applied 数量：`{forecast.get('calibration_applied_count', 0)}`",
            f"- missing forecast 数量：`{forecast.get('missing_forecast_count', 0)}`",
            f"- locations_pending.csv 路径：`{forecast.get('locations_pending_path', 'data/weather/locations_pending.csv')}`",
            f"- 待补 station/location 条目数：`{forecast.get('locations_pending_count', 0)}`",
            "",
            "### forecast_source 分布",
            "",
            "| forecast_source | 数量 |",
            "|---|---:|",
        ]
    )
    for source, count in forecast.get("forecast_source_counts", {}).items():
        lines.append(f"| {source} | {count} |")
    if not forecast.get("forecast_source_counts"):
        lines.append("| 无 | 0 |")
    lines.extend(["", "### std_method 分布", "", "| std_method | 数量 |", "|---|---:|"])
    for method, count in forecast.get("std_method_counts", {}).items():
        lines.append(f"| {method} | {count} |")
    if not forecast.get("std_method_counts"):
        lines.append("| 无 | 0 |")
    lines.extend(["", "### calibration_quality 分布", "", "| calibration_quality | 数量 |", "|---|---:|"])
    for quality, count in forecast.get("calibration_quality_counts", {}).items():
        lines.append(f"| {quality} | {count} |")
    if not forecast.get("calibration_quality_counts"):
        lines.append("| 无 | 0 |")

    lines.extend(
        [
            "",
            "## 8. 结论和下一步建议",
            "",
        ]
    )
    for recommendation in result.get("recommendations", []):
        lines.append(f"- {recommendation}")
    if not result.get("recommendations"):
        lines.append("- 继续每天一次记录，先积累 forward replay 样本。")
    lines.append("")
    lines.append("本报告只做诊断，不改变策略阈值，不放宽 strict gate，也不是历史回测。")
    return "\n".join(lines) + "\n"


def _recent_day_dirs(root: Path, days: int) -> list[Path]:
    if not root.exists():
        return []
    day_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)
    ]
    return sorted(day_dirs, key=lambda path: path.name)[-max(days, 0) :]


def _summary_files(day_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for day_dir in day_dirs:
        files.extend(sorted(day_dir.glob("capture_summary_*.json")))
    return sorted(files)


def _missing_forecast_lookup_keys(
    signals: list[dict[str, Any]],
    summary: dict[str, Any],
) -> set[str]:
    keys: set[str] = set()
    for item in summary.get("skipped") or []:
        text = str(item)
        if "MISSING_FORECAST_DATA" in text:
            parts = text.split(":", 2)
            if len(parts) == 3:
                keys.add(parts[2])
    for signal in signals:
        if signal.get("signal_status") == "MISSING_FORECAST_DATA":
            key = (
                f"{signal.get('location_name') or 'unknown'}:"
                f"{signal.get('target_date') or 'unknown'}:{signal.get('metric') or 'unknown'}"
            )
            keys.add(key)
    return keys


def _station_details_for_candidate(candidate: dict[str, Any]):
    text = " ".join(
        str(candidate.get(key) or "")
        for key in [
            "resolution_source",
            "resolution_text_excerpt",
            "question",
            "slug",
        ]
    )
    return extract_weather_station_details(text)


def _candidate_resolution_source(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("resolution_source") or candidate.get("resolution_text_excerpt")
    return str(value) if value else None


def _rough_location_from_question(question: str) -> str | None:
    for pattern in [r"\bin\s+(.+?)\s+be\b", r"\bin\s+(.+?)\s+(?:on|by|before|after)\b"]:
        match = re.search(pattern, question, flags=re.I)
        if match:
            return " ".join(match.group(1).strip(" ?.,;:\"'").split())
    return None


def _missing_forecast_reason(
    *,
    candidate: dict[str, Any],
    classification: Any,
    resolver: LocationResolver,
    captured_at: str,
) -> str:
    if classification.metric not in {"high_temperature", "low_temperature", "average_temperature"}:
        return "metric_not_supported"
    if not classification.target_date or not classification.location_name:
        return "parse_error"
    station = _station_details_for_candidate(candidate)
    if station.station_id and resolver.resolve_station(station.station_id) is None:
        return "station_not_in_locations_csv"
    if resolver.resolve(classification.location_name) is None:
        return "location_not_in_locations_csv"
    captured_date = _date_from_timestamp(captured_at)
    if captured_date is not None and classification.target_date < captured_date:
        return "target_date_out_of_provider_range"
    return "unknown"


def _date_from_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value) else None


def _write_locations_pending(records: list[dict[str, Any]], output_path: str | Path) -> int:
    path = Path(output_path)
    fields = [
        "source",
        "detected_location_name",
        "detected_station_id",
        "detected_station_name",
        "country_hint",
        "example_market_id",
        "example_slug",
        "example_question",
        "resolution_source",
        "suggested_latitude",
        "suggested_longitude",
        "suggested_action",
        "status",
    ]
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = (
                    str(row.get("detected_location_name") or ""),
                    str(row.get("detected_station_id") or ""),
                )
                pending[key] = {field: row.get(field, "") for field in fields}
    added_or_updated = 0
    for record in records:
        if record.get("reason") not in {"location_not_in_locations_csv", "station_not_in_locations_csv"}:
            continue
        location = str(record.get("parsed_location_name") or "")
        station_id = str(record.get("extracted_station_id") or "")
        key = (location, station_id)
        row = {
            "source": "daily_capture_missing_forecast",
            "detected_location_name": location,
            "detected_station_id": station_id,
            "detected_station_name": record.get("extracted_station_name") or "",
            "country_hint": "",
            "example_market_id": record.get("market_id") or "",
            "example_slug": record.get("slug") or "",
            "example_question": record.get("question") or "",
            "resolution_source": record.get("resolution_source") or "",
            "suggested_latitude": "",
            "suggested_longitude": "",
            "suggested_action": "add verified station coordinates to data/weather/locations.csv",
            "status": "pending_manual_coordinates",
        }
        if pending.get(key) != row:
            pending[key] = row
            added_or_updated += 1
    if pending or added_or_updated:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in pending.values():
                writer.writerow(row)
    return added_or_updated


def _missing_forecast_suggested_action(key: str, records: list[dict[str, Any]]) -> str:
    reasons = {str(record.get("reason")) for record in records if key in {str(record.get("extracted_station_id") or ""), str(record.get("parsed_location_name") or "")}}
    if "station_not_in_locations_csv" in reasons:
        return "人工补 station_id 对应机场站点经纬度"
    if "location_not_in_locations_csv" in reasons:
        return "人工补 location 经纬度或 station mapping"
    if "target_date_out_of_provider_range" in reasons:
        return "确认 provider 日期窗口；过期市场不需要 current forecast"
    if "parse_error" in reasons:
        return "补 classifier 样例或人工确认问题格式"
    return "检查 provider error / forecast cache"


def _missing_reason_zh(reason: str) -> str:
    translations = {
        "location_not_in_locations_csv": "地点不在 locations.csv，无法取得经纬度。",
        "station_not_in_locations_csv": "结算站点不在 locations.csv，不能按 station-level 拉 forecast。",
        "target_date_out_of_provider_range": "目标日期不在当前 forecast provider 可用窗口。",
        "metric_not_supported": "当前只支持高温、低温、平均温度。",
        "provider_error": "provider 请求或解析失败。",
        "parse_error": "问题文本未能稳定解析出地点/日期/阈值。",
        "unknown": "现有 JSON 不足以判断具体原因。",
    }
    return translations.get(reason, "未知原因。")



def _load_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_json_list(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as file:
        return max(0, sum(1 for _ in csv.DictReader(file)))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _locations_suggestions_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "total": 0,
            "high_confidence_promotable": 0,
            "suggested_locations": [],
        }
    rows = _read_csv_rows(path)
    high_confidence = [
        row
        for row in rows
        if row.get("status") == "suggested" and (_float_or_none(row.get("confidence")) or 0) >= 0.85
    ]
    return {
        "exists": True,
        "path": str(path),
        "total": len(rows),
        "high_confidence_promotable": len(high_confidence),
        "suggested_locations": [row.get("detected_location_name") for row in high_confidence[:20]],
    }


def _resolve_artifact_path(value: Any, run_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    candidate = run_dir / path.name
    return candidate if candidate.exists() else path


def _run_row(
    summary: dict[str, Any],
    summary_path: Path,
    strategy_candidates: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    captured_at = str(summary.get("captured_at") or "")
    return {
        "date": summary_path.parent.name,
        "summary_path": str(summary_path),
        "captured_at": captured_at,
        "strategy_candidates_count": int(summary.get("strategy_candidates_count") or len(strategy_candidates)),
        "weather_candidates_count": int(summary.get("weather_candidates_count") or 0),
        "weather_alpha_signals_count": int(summary.get("weather_alpha_signals_count") or len(signals)),
        "snapshots_inserted": int(summary.get("snapshots_inserted") or 0),
        "backtest_saved": int(summary.get("backtest_saved") or 0),
        "backtest_skipped": int(summary.get("backtest_skipped") or 0),
        "errors": list(summary.get("errors") or []),
        "warnings": list(summary.get("warnings") or []) + list(summary.get("timing_warnings") or []),
    }


def _record_signal_diagnostics(
    signal: dict[str, Any],
    *,
    edge_bucket_counts: Counter[str],
    edge_records: list[dict[str, Any]],
    forecast_source_counts: Counter[str],
    std_method_counts: Counter[str],
    calibration_quality_counts: Counter[str],
    source_location_counts: Counter[str],
    timing_issue_counts: Counter[str],
) -> None:
    yes_edge = _float_or_none(signal.get("yes_model_edge"))
    no_edge = _float_or_none(signal.get("no_model_edge"))
    dominant_edge = max(value for value in [yes_edge, no_edge] if value is not None) if yes_edge is not None or no_edge is not None else None
    if dominant_edge is not None:
        edge_bucket_counts[_edge_bucket(dominant_edge)] += 1
    edge_records.append(
        {
            "market_id": signal.get("market_id"),
            "question": signal.get("question"),
            "model_p_yes": signal.get("model_p_yes"),
            "yes_breakeven": signal.get("yes_breakeven"),
            "no_upper_bound": signal.get("no_upper_bound"),
            "yes_model_edge": yes_edge,
            "no_model_edge": no_edge,
            "suggested_paper_side": signal.get("suggested_paper_side"),
            "edge_threshold": signal.get("edge_threshold"),
        }
    )
    forecast_source_counts[str(signal.get("forecast_source") or "unknown")] += 1
    std_method_counts[str(signal.get("std_method") or "unknown")] += 1
    calibration_quality_counts[str(signal.get("calibration_quality") or "unknown")] += 1
    if signal.get("source_location_name"):
        source_location_counts[str(signal.get("source_location_name"))] += 1
    for warning in _signal_warnings(signal):
        if "forecast_issued" in warning or "as_of_time" in warning or "forecast_after" in warning:
            timing_issue_counts[warning] += 1


def _candidate_failure_reasons(
    candidate: dict[str, Any],
    classification: Any,
    signal_by_market_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    inferred: list[str] = []
    if classification.market_type not in SUPPORTED_WEATHER_TYPES:
        if _weather_looking_candidate(candidate):
            reasons.append("unsupported_weather_market_type")
        else:
            reasons.append("not_weather_candidate")
        return reasons, inferred
    if not classification.location_name:
        reasons.append("missing_location")
    if not classification.target_date:
        reasons.append("missing_target_date")
    if _float_or_none(candidate.get("yes_executable_avg_buy_price")) is None or _float_or_none(
        candidate.get("no_executable_avg_buy_price")
    ) is None:
        reasons.append("missing_orderbook")
    if any("insufficient" in str(reason) for reason in candidate.get("reasons") or []):
        reasons.append("insufficient_depth")
    signal = signal_by_market_id.get(str(candidate.get("market_id")))
    if signal is None:
        reasons.append("missing_forecast")
        inferred.append("missing_forecast")
        return _unique(reasons), _unique(inferred)
    if signal.get("model_p_yes") is None:
        reasons.append("no_model_probability")
    side = str(signal.get("suggested_paper_side") or "UNKNOWN")
    if side == "NONE":
        reasons.append("no_paper_side")
        yes_edge = _float_or_none(signal.get("yes_model_edge"))
        no_edge = _float_or_none(signal.get("no_model_edge"))
        threshold = _float_or_none(signal.get("edge_threshold")) or 0.05
        if (yes_edge is None or yes_edge < threshold) and (no_edge is None or no_edge < threshold):
            reasons.append("edge_below_threshold")
            inferred.append("edge_below_threshold")
    if str(signal.get("signal_status") or "") != "VALID":
        reasons.append("validation_not_valid")
    warnings = _signal_warnings(signal)
    if any("station" in warning and ("mismatch" in warning or "missing" in warning) for warning in warnings):
        reasons.append("station_not_matched")
    if (classification.market_type == MarketType.weather_temperature_exact_bucket) and not bool(
        signal.get("bucket_numeric_boundary_confirmed")
    ):
        reasons.append("bucket_boundary_not_confirmed")
    if (classification.market_type == MarketType.weather_temperature_range_bucket) and not bool(
        signal.get("range_numeric_boundary_confirmed")
    ):
        reasons.append("bucket_boundary_not_confirmed")
    if str(signal.get("ambiguity_risk") or "").upper() in {"MEDIUM", "HIGH"} or str(
        signal.get("dispute_risk") or ""
    ).upper() in {"MEDIUM", "HIGH"}:
        reasons.append("resolution_risk_not_low")
    if any("forecast_issued" in warning or "as_of_time" in warning or "forecast_after" in warning for warning in warnings):
        reasons.append("forecast_timing_issue")
    return _unique(reasons) or ["unknown_reason"], _unique(inferred)


def _market_type_bucket(candidate: dict[str, Any], classification: Any) -> str:
    question = str(candidate.get("question") or "")
    if classification.market_type == MarketType.weather_temperature_exact_bucket:
        return "exact_temperature_bucket"
    if classification.market_type == MarketType.weather_temperature_range_bucket:
        return "range_bucket"
    if classification.market_type == MarketType.weather_temperature_threshold:
        return "threshold_above_below"
    if _weather_looking_candidate(candidate):
        return "unsupported_wording"
    if re.search(r"\bbe\s+\d+(?:\.\d+)?\s*(?:°?\s*[CF]|Celsius|Fahrenheit)\b", question, re.I):
        return "exact_temperature_bucket"
    if re.search(r"\b(above|below|over|under|or below|at least|at most)\b", question, re.I):
        return "threshold_above_below"
    return "unknown"


def _weather_looking_candidate(candidate: dict[str, Any]) -> bool:
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ["question", "slug", "category"]
    ).casefold()
    return any(
        marker in text
        for marker in [
            "weather",
            "temperature",
            "highest temperature",
            "lowest temperature",
            "rainfall",
            "hurricane",
            "snow",
            "°c",
            "°f",
            " c ",
            " f ",
        ]
    )


def _funnel_rows(totals: dict[str, int]) -> list[dict[str, Any]]:
    strategy = int(totals.get("strategy_candidates") or 0)
    weather = int(totals.get("weather_candidates") or 0)
    alpha = int(totals.get("weather_alpha_signals") or 0)
    paper_side = int(totals.get("paper_side_yes_no") or 0)
    strict_saved = int(totals.get("strict_saved") or 0)
    return [
        {
            "stage": "strategy candidates",
            "count": strategy,
            "conversion_rate": "100.0%" if strategy else "n/a",
            "explanation": "本次 strategy scan 找到的候选市场总数。",
        },
        {
            "stage": "weather candidates",
            "count": weather,
            "conversion_rate": _pct(weather, strategy),
            "explanation": "被识别为天气相关的候选市场。",
        },
        {
            "stage": "supported weather market / alpha signals",
            "count": alpha,
            "conversion_rate": _pct(alpha, weather),
            "explanation": "成功生成模型概率和 edge 的天气信号。",
        },
        {
            "stage": "paper side YES/NO",
            "count": paper_side,
            "conversion_rate": _pct(paper_side, alpha),
            "explanation": "模型 edge 超过阈值并给出纸面方向的信号。",
        },
        {
            "stage": "strict saved",
            "count": strict_saved,
            "conversion_rate": _pct(strict_saved, paper_side),
            "explanation": "通过 strict gate 写入 forward replay paper dataset。",
        },
    ]


def _top_bottlenecks(
    failure_reasons: Counter[str],
    skipped_reasons: Counter[str],
    *,
    totals: dict[str, int],
    station_mismatch: int,
) -> list[dict[str, Any]]:
    combined = Counter(failure_reasons)
    combined.update(skipped_reasons)
    if int(totals.get("weather_candidates") or 0) > 0:
        combined.pop("not_weather_candidate", None)
    if station_mismatch:
        combined["station_not_matched"] += station_mismatch
    if int(totals.get("weather_alpha_signals") or 0) == 0 and int(totals.get("weather_candidates") or 0) > 0:
        combined["missing_forecast"] += int(totals.get("weather_candidates") or 0)
    return [
        {"reason": reason, "count": count, "explanation": _reason_zh(reason)}
        for reason, count in combined.most_common(3)
    ]


def _recommendations(
    totals: dict[str, int],
    failure_reasons: Counter[str],
    station_mismatch: int,
) -> list[str]:
    recommendations: list[str] = []
    strategy = int(totals.get("strategy_candidates") or 0)
    weather = int(totals.get("weather_candidates") or 0)
    alpha = int(totals.get("weather_alpha_signals") or 0)
    paper_side = int(totals.get("paper_side_yes_no") or 0)
    strict_saved = int(totals.get("strict_saved") or 0)
    if weather <= max(1, strategy // 20):
        recommendations.append("weather_candidates 很少：建议 daily capture 使用 --limit 100 或更高；可降低 min-liquidity 做观察扫描，但不要用于交易。")
        recommendations.append("如果天气市场长期稀少，考虑增加 weather-specific scan 模式。")
    if weather > 0 and alpha == 0:
        recommendations.append("weather_candidates 多但 alpha signals 少：优先检查 market_type_classifier 和 forecast/location 匹配，不要先调低 edge_threshold。")
        recommendations.append("输出 unsupported weather market examples 后，再决定是否扩展 classifier 支持更多天气问题格式。")
    if alpha > 0 and paper_side == 0:
        recommendations.append("alpha signals 有但 no_paper_side 多：先观察 edge 分布；不要马上降低阈值。")
    if station_mismatch or failure_reasons.get("station_not_matched", 0):
        recommendations.append("station mismatch 较多：扩展 data/weather/locations.csv，把 Polymarket resolution station 映射到站点坐标。")
        recommendations.append("不要忽略 station mismatch，否则 forecast 与结算口径可能不一致。")
    if failure_reasons.get("bucket_boundary_not_confirmed", 0):
        recommendations.append("bucket boundary 未确认较多：保持 NEEDS_MANUAL_REVIEW，不要 strict 保存；可做非 strict observation dataset。")
    if failure_reasons.get("validation_not_valid", 0) or failure_reasons.get("resolution_risk_not_low", 0):
        recommendations.append("validation 风险较多：优先看 resolution/source/station，而不是改策略阈值。")
    if strict_saved > 0:
        recommendations.append("已有 signal 进入 paper dataset：后续等市场结算后用 weather-backtest resolve 评估 PnL 和 Brier。")
    if not recommendations:
        recommendations.append("继续每天一次记录，先积累 forward replay 数据。")
    return _unique(recommendations)


def _counter_from_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, count in value.items():
            try:
                counter[str(key)] += int(count)
            except (TypeError, ValueError):
                counter[str(key)] += 1
    return counter


def _signal_warnings(signal: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ["warnings", "validation_warnings", "data_provenance_warnings"]:
        value = signal.get(key)
        if isinstance(value, list):
            warnings.extend(str(item) for item in value)
    return _unique(warnings)


def _has_any_warning(signal: dict[str, Any], needles: set[str]) -> bool:
    warnings = _signal_warnings(signal)
    return any(warning in needles for warning in warnings)


def _edge_bucket(value: float) -> str:
    for label, lower, upper in EDGE_BUCKETS:
        if lower <= value < upper:
            return label
    return ">5%"


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def _reason_zh(reason: str) -> str:
    return REASON_ZH.get(reason) or SKIPPED_REASON_ZH.get(reason) or "暂无内置解释。"


def _join_or_none(values: Any) -> str:
    if not values:
        return "无"
    if isinstance(values, list):
        return "<br>".join(str(item) for item in values) if values else "无"
    return str(values)


def _fmt_float(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.4f}"


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
