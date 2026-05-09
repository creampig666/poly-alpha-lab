"""CLI entry point for poly-alpha-lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from poly_alpha_lab.client import GammaClient
from poly_alpha_lab.clob_client import ClobClient
from poly_alpha_lab.config import settings
from poly_alpha_lab.daily_capture import (
    DailyWeatherCaptureConfig,
    daily_capture_summary_to_markdown,
    run_daily_weather_capture,
)
from poly_alpha_lab.filters import filter_markets
from poly_alpha_lab.journal import JournalEntry, ResearchJournal
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.markdown_report import (
    candidates_report,
    market_report,
    markets_report,
    resolution_analysis_report,
)
from poly_alpha_lab.models import Market, MarketStructureError
from poly_alpha_lab.network_debug import (
    network_debug_report_to_markdown,
    run_network_debug,
    write_network_debug_report,
)
from poly_alpha_lab.resolution_analyzer import analyze_resolution as analyze_market_resolution
from poly_alpha_lab.strategy_runner import (
    scan_strategy_candidates,
    strategy_candidates_report,
    write_strategy_candidates_json,
)
from poly_alpha_lab.weather_alpha import (
    run_weather_alpha_scan,
    weather_alpha_report,
    write_weather_alpha_signals_json,
)
from poly_alpha_lab.weather_backtest import WeatherBacktestSnapshot, WeatherBacktestStore
from poly_alpha_lab.weather_calibration import (
    WeatherCalibrationSummary,
    fit_weather_calibration,
    load_calibration_summaries,
    normalize_group_by,
    write_calibration_csv,
    write_calibration_json,
)
from poly_alpha_lab.weather_data import (
    CsvWeatherDataProvider,
    LocationResolver,
    OpenMeteoForecastProvider,
)
from poly_alpha_lab.weather_dataset_builder import (
    OpenMeteoHistoricalDatasetProvider,
    build_provider_semantics_audit,
    build_weather_dataset,
    debug_open_meteo_provider,
    manual_validation_to_markdown,
    parse_csv_list,
    provider_semantics_audit_to_markdown,
    provider_debug_report_to_markdown,
    validate_manual_forecast_actual_csv,
    weather_dataset_summary_to_markdown,
    write_data_source_options_markdown,
    write_manual_forecast_actual_template,
    write_provider_semantics_audit,
    write_provider_debug_report,
)
from poly_alpha_lab.weather_model_diagnostics import (
    diagnose_weather_models,
    weather_model_diagnostics_report,
    write_weather_model_diagnostics_csv,
)


def fetch_binary_liquidity(clob_client: ClobClient, market: Market) -> BinaryMarketLiquidity | None:
    """Fetch YES and NO token books; return None when the market cannot be mapped."""

    try:
        yes_token_id = market.yes_token_id
        no_token_id = market.no_token_id
    except MarketStructureError:
        return None

    try:
        yes_book = clob_client.get_orderbook(yes_token_id)
        no_book = clob_client.get_orderbook(no_token_id)
    except Exception:
        return None

    return BinaryMarketLiquidity(yes_book=yes_book, no_book=no_book)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poly-alpha-lab",
        description="Read-only Polymarket Gamma API market research CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Fetch and analyze a list of active markets.")
    scan.add_argument("--mode", choices=["candidates", "ev"], default="candidates")
    scan.add_argument("--limit", type=int, default=settings.default_limit)
    scan.add_argument("--offset", type=int, default=None)
    scan.add_argument("--min-liquidity", type=float, default=settings.min_liquidity)
    scan.add_argument("--fair-yes", type=float, default=None)
    scan.add_argument("--size", type=float, default=settings.default_position_size)
    scan.add_argument("--min-net-edge", type=float, default=settings.min_net_edge)
    scan.add_argument("--order", default="volume")

    market = subparsers.add_parser("market", help="Fetch and analyze one Gamma market by ID.")
    market.add_argument("market_id")
    market.add_argument("--fair-yes", type=float, default=settings.default_fair_yes_probability)
    market.add_argument("--size", type=float, default=settings.default_position_size)
    market.add_argument("--min-net-edge", type=float, default=settings.min_net_edge)

    slug = subparsers.add_parser("slug", help="Fetch and analyze one Gamma market by slug.")
    slug.add_argument("slug")
    slug.add_argument("--fair-yes", type=float, default=settings.default_fair_yes_probability)
    slug.add_argument("--size", type=float, default=settings.default_position_size)
    slug.add_argument("--min-net-edge", type=float, default=settings.min_net_edge)

    analyze_resolution = subparsers.add_parser(
        "analyze-resolution",
        help="Analyze resolution criteria risk for one market.",
    )
    analyze_target = analyze_resolution.add_mutually_exclusive_group(required=True)
    analyze_target.add_argument("--market-id")
    analyze_target.add_argument("--slug")

    strategy = subparsers.add_parser("strategy", help="Run paper strategy helpers.")
    strategy_subparsers = strategy.add_subparsers(dest="strategy_command", required=True)
    strategy_scan = strategy_subparsers.add_parser(
        "scan",
        help="Scan breakeven paper strategy candidates.",
    )
    strategy_scan.add_argument("--limit", type=int, default=settings.default_limit)
    strategy_scan.add_argument("--min-liquidity", type=float, default=settings.min_liquidity)
    strategy_scan.add_argument("--size", type=float, default=settings.default_position_size)
    strategy_scan.add_argument("--min-grade", choices=["A", "B", "C"], default="B")
    strategy_scan.add_argument(
        "--max-resolution-risk",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    strategy_scan.add_argument("--category")
    strategy_scan.add_argument("--include-long-dated", action="store_true", default=True)
    strategy_scan.add_argument("--include-longshots", action="store_true")
    strategy_scan.add_argument("--output-json")

    alpha = subparsers.add_parser("alpha", help="Run paper alpha modules.")
    alpha_subparsers = alpha.add_subparsers(dest="alpha_command", required=True)
    weather_scan = alpha_subparsers.add_parser(
        "scan-weather",
        help="Run paper weather temperature threshold alpha from strategy JSON.",
    )
    weather_scan.add_argument("--strategy-json", required=True)
    weather_scan.add_argument("--weather-provider", choices=["csv", "open-meteo"], default="csv")
    weather_scan.add_argument("--weather-data", default="data/weather/weather_forecasts.csv")
    weather_scan.add_argument("--locations-file", default="data/weather/locations.csv")
    weather_scan.add_argument("--fallback-forecast-std", type=float)
    weather_scan.add_argument("--cache-dir", default="data/weather/cache")
    weather_scan.add_argument("--refresh-cache", action="store_true")
    weather_scan.add_argument("--edge-threshold", type=float, default=0.05)
    weather_scan.add_argument("--bucket-mode", choices=["rounded", "floor"], default="rounded")
    weather_scan.add_argument(
        "--weather-model",
        choices=["normal", "student_t", "normal_mixture"],
        default="normal",
    )
    weather_scan.add_argument("--student-t-df", type=float, default=5)
    weather_scan.add_argument("--mixture-tail-weight", type=float, default=0.10)
    weather_scan.add_argument("--mixture-tail-scale", type=float, default=2.5)
    weather_scan.add_argument("--calibration-json")
    weather_scan.add_argument("--calibration-group", default="metric,horizon_bucket")
    weather_scan.add_argument("--use-calibrated-std", action="store_true")
    weather_scan.add_argument("--use-calibrated-bias", action="store_true")
    weather_scan.add_argument("--min-calibration-samples", type=int, default=30)
    weather_scan.add_argument("--allow-low-quality-calibration", action="store_true")
    weather_scan.add_argument("--as-of-time")
    weather_scan.add_argument("--output-json")

    diagnose_weather = alpha_subparsers.add_parser(
        "diagnose-weather-models",
        help="Diagnose exact temperature bucket probabilities across weather models.",
    )
    diagnose_weather.add_argument("--mean", type=float, required=True)
    diagnose_weather.add_argument("--std", type=float, required=True)
    diagnose_weather.add_argument("--unit", choices=["C", "F"], default="C")
    diagnose_weather.add_argument("--bucket-mode", choices=["rounded", "floor"], default="rounded")
    diagnose_weather.add_argument("--student-t-df", type=float, default=5)
    diagnose_weather.add_argument("--mixture-tail-weight", type=float, default=0.10)
    diagnose_weather.add_argument("--mixture-tail-scale", type=float, default=2.5)
    diagnose_weather.add_argument("--k-min", type=int, default=18)
    diagnose_weather.add_argument("--k-max", type=int, default=30)
    diagnose_weather.add_argument("--output-csv", required=True)

    weather_backtest = subparsers.add_parser(
        "weather-backtest",
        help="Manage weather forward replay snapshots.",
    )
    weather_backtest_subparsers = weather_backtest.add_subparsers(
        dest="weather_backtest_command",
        required=True,
    )
    add_signals = weather_backtest_subparsers.add_parser(
        "add-from-signals",
        help="Add paper replay snapshots from weather alpha signals JSON.",
    )
    add_signals.add_argument("--signals-json", required=True)
    add_signals.add_argument("--entry-size", type=float, required=True)
    add_signals.add_argument("--strict", action="store_true")
    add_signals.add_argument("--include-needs-review", action="store_true")
    add_signals.add_argument("--allow-unconfirmed-bucket", action="store_true")
    add_signals.add_argument("--allow-station-mismatch", action="store_true")
    add_signals.add_argument("--allow-sample-data", action="store_true")
    add_signals.add_argument("--status", choices=["OPEN", "SKIPPED"])

    backtest_list = weather_backtest_subparsers.add_parser(
        "list",
        help="List recent weather replay snapshots.",
    )
    backtest_list.add_argument("--limit", type=int, default=20)
    backtest_list.add_argument("--status", choices=["OPEN", "RESOLVED", "VOID", "SKIPPED"])
    backtest_list.add_argument("--signal-status")
    backtest_list.add_argument("--side", choices=["YES", "NO", "NONE"])
    backtest_list.add_argument("--location")

    backtest_resolve = weather_backtest_subparsers.add_parser(
        "resolve",
        help="Resolve one weather replay snapshot.",
    )
    backtest_resolve.add_argument("--id", type=int, required=True)
    backtest_resolve.add_argument("--actual-value", type=float)
    backtest_resolve.add_argument("--resolution-value", type=int, choices=[0, 1], required=True)
    backtest_resolve.add_argument("--notes")

    weather_backtest_subparsers.add_parser(
        "summary",
        help="Summarize weather replay performance.",
    )

    backtest_export = weather_backtest_subparsers.add_parser(
        "export",
        help="Export weather replay snapshots to CSV.",
    )
    backtest_export.add_argument("--output", required=True)

    daily_capture = subparsers.add_parser(
        "daily-capture",
        help="Run forward replay capture jobs.",
    )
    daily_capture_subparsers = daily_capture.add_subparsers(
        dest="daily_capture_command",
        required=True,
    )
    daily_weather = daily_capture_subparsers.add_parser(
        "weather",
        help="Capture weather strategy, alpha, CLOB snapshots, and paper replay rows.",
    )
    daily_weather.add_argument("--limit", type=int, default=100)
    daily_weather.add_argument("--min-liquidity", type=float, default=1000)
    daily_weather.add_argument("--size", type=float, default=10)
    daily_weather.add_argument("--weather-provider", choices=["csv", "open-meteo"], default="csv")
    daily_weather.add_argument("--weather-data")
    daily_weather.add_argument("--calibration-json")
    daily_weather.add_argument(
        "--weather-model",
        choices=["normal", "student_t", "normal_mixture"],
        default="normal",
    )
    daily_weather.add_argument("--bucket-mode", choices=["rounded", "floor"], default="rounded")
    daily_weather.add_argument("--proxy")
    daily_trust_group = daily_weather.add_mutually_exclusive_group()
    daily_trust_group.add_argument("--trust-env", dest="trust_env", action="store_true", default=True)
    daily_trust_group.add_argument("--no-trust-env", dest="trust_env", action="store_false")
    daily_weather.add_argument("--output-dir", default="data/daily")
    daily_weather.add_argument("--snapshot-db", default="data/polymarket_snapshots.sqlite")
    daily_weather.add_argument("--backtest-db", default="data/weather_backtest.sqlite")
    daily_weather.add_argument("--entry-size", type=float, default=10)
    daily_weather.add_argument("--strict", action="store_true")
    daily_weather.add_argument("--include-needs-review", action="store_true")
    daily_weather.add_argument("--dry-run", action="store_true")
    daily_weather.add_argument("--sizes", default="10,50,100")
    daily_weather.add_argument("--forecast-time-tolerance-seconds", type=float, default=120)

    weather_calibration = subparsers.add_parser(
        "weather-calibration",
        help="Fit weather forecast error calibration summaries.",
    )
    weather_calibration_subparsers = weather_calibration.add_subparsers(
        dest="weather_calibration_command",
        required=True,
    )
    calibration_fit = weather_calibration_subparsers.add_parser(
        "fit",
        help="Fit calibration summaries from forecast-vs-actual CSV history.",
    )
    calibration_fit.add_argument("--input", required=True)
    calibration_fit.add_argument("--output-json", required=True)
    calibration_fit.add_argument("--output-csv", required=True)
    calibration_fit.add_argument("--group-by", default="metric,horizon_bucket")
    calibration_fit.add_argument("--min-samples", type=int, default=20)
    calibration_fit.add_argument("--bias-shrinkage-k", type=float, default=30)

    weather_dataset = subparsers.add_parser(
        "weather-dataset",
        help="Build historical weather forecast-vs-actual calibration datasets.",
    )
    weather_dataset_subparsers = weather_dataset.add_subparsers(
        dest="weather_dataset_command",
        required=True,
    )
    dataset_build = weather_dataset_subparsers.add_parser(
        "build",
        help="Build forecast_actual_history.csv for weather calibration.",
    )
    dataset_build.add_argument("--locations-file", default="data/weather/locations.csv")
    dataset_build.add_argument("--output", default="data/weather/forecast_actual_history.csv")
    dataset_build.add_argument("--provider", choices=["open-meteo"], default="open-meteo")
    dataset_build.add_argument("--start-date", required=True)
    dataset_build.add_argument("--end-date", required=True)
    dataset_build.add_argument("--metrics", default="high_temperature,low_temperature")
    dataset_build.add_argument("--forecast-issue-hours", default="0,6,12,18")
    dataset_build.add_argument("--horizons", default="12,24,48")
    dataset_build.add_argument("--cache-dir", default="data/weather/dataset_cache")
    dataset_build.add_argument("--refresh-cache", action="store_true")
    dataset_build.add_argument("--audit-output", default="data/weather/provider_semantics_audit.json")
    dataset_build.add_argument("--debug-provider", action="store_true")
    dataset_build.add_argument("--timeout-seconds", type=float, default=15)
    dataset_build.add_argument("--print-request-url", action="store_true")
    dataset_build.add_argument("--proxy")
    trust_group = dataset_build.add_mutually_exclusive_group()
    trust_group.add_argument("--trust-env", dest="trust_env", action="store_true", default=True)
    trust_group.add_argument("--no-trust-env", dest="trust_env", action="store_false")

    debug_provider = weather_dataset_subparsers.add_parser(
        "debug-provider",
        help="Debug one weather data provider request pair.",
    )
    debug_provider.add_argument("--provider", choices=["open-meteo"], default="open-meteo")
    debug_provider.add_argument("--location", required=True)
    debug_provider.add_argument("--latitude", type=float, required=True)
    debug_provider.add_argument("--longitude", type=float, required=True)
    debug_provider.add_argument("--target-date", required=True)
    debug_provider.add_argument("--forecast-issued-at", required=True)
    debug_provider.add_argument("--metric", required=True)
    debug_provider.add_argument("--horizon", type=float, required=True)
    debug_provider.add_argument("--cache-dir", default="data/weather/debug_cache")
    debug_provider.add_argument("--refresh-cache", action="store_true")
    debug_provider.add_argument("--timeout-seconds", type=float, default=30)
    debug_provider.add_argument("--print-request-url", action="store_true")
    debug_provider.add_argument("--proxy")
    debug_trust_group = debug_provider.add_mutually_exclusive_group()
    debug_trust_group.add_argument("--trust-env", dest="trust_env", action="store_true", default=True)
    debug_trust_group.add_argument("--no-trust-env", dest="trust_env", action="store_false")
    debug_provider.add_argument("--output", default="data/weather/provider_debug_report.json")

    debug_network = weather_dataset_subparsers.add_parser(
        "debug-network",
        help="Diagnose DNS/TCP/HTTP connectivity to a weather provider URL.",
    )
    debug_network.add_argument(
        "--url",
        default="https://historical-forecast-api.open-meteo.com/v1/forecast",
    )
    debug_network.add_argument("--timeout-seconds", type=float, default=30)
    debug_network.add_argument("--proxy")
    net_trust_group = debug_network.add_mutually_exclusive_group()
    net_trust_group.add_argument("--trust-env", dest="trust_env", action="store_true", default=True)
    net_trust_group.add_argument("--no-trust-env", dest="trust_env", action="store_false")
    debug_network.add_argument("--print-env-proxy", action="store_true")
    debug_network.add_argument("--output", default="data/weather/network_debug_report.json")

    manual_template = weather_dataset_subparsers.add_parser(
        "manual-template",
        help="Create an empty manual verified forecast/actual CSV template.",
    )
    manual_template.add_argument("--locations-file", default="data/weather/locations.csv")
    manual_template.add_argument("--output", default="data/weather/forecast_actual_history_manual_template.csv")
    manual_template.add_argument("--rows-per-location", type=int, default=2)

    validate_manual = weather_dataset_subparsers.add_parser(
        "validate-manual-csv",
        help="Validate a manual forecast/actual CSV before calibration.",
    )
    validate_manual.add_argument("--input", required=True)

    data_source_options = weather_dataset_subparsers.add_parser(
        "data-source-options",
        help="Write weather data source options assessment markdown.",
    )
    data_source_options.add_argument("--output", default="data/weather/data_source_options.md")
    data_source_options.add_argument("--open-meteo-status", default="UNKNOWN")
    data_source_options.add_argument("--failure-reason", default="")

    journal = subparsers.add_parser("journal", help="Manage the local research journal.")
    journal_subparsers = journal.add_subparsers(dest="journal_command", required=True)

    add = journal_subparsers.add_parser("add", help="Add a manual paper trade or research record.")
    add.add_argument("--from-json-file")
    add.add_argument("--market-id")
    add.add_argument("--slug")
    add.add_argument("--question")
    add.add_argument("--category")
    add.add_argument("--end-date")
    add.add_argument("--candidate-score", type=float)
    add.add_argument("--candidate-grade", choices=["A", "B", "C", "SKIP"])
    add.add_argument("--side", choices=["YES", "NO", "NONE"])
    add.add_argument("--fair-yes", type=float)
    add.add_argument(
        "--probability-source",
        choices=[
            "manual",
            "external_model",
            "ai_estimate",
            "weather_threshold_model",
            "unknown",
        ],
    )
    add.add_argument("--entry-price", type=float)
    add.add_argument("--entry-size", type=float)
    add.add_argument("--fee-per-share", type=float)
    add.add_argument("--expected-value-per-share", type=float)
    add.add_argument("--expected-profit", type=float)
    add.add_argument("--rationale")
    add.add_argument("--status", choices=["OPEN", "RESOLVED", "VOID", "SKIPPED"])
    add.add_argument("--notes")

    list_cmd = journal_subparsers.add_parser("list", help="List recent journal entries.")
    list_cmd.add_argument("--limit", type=int, default=20)
    list_cmd.add_argument("--status", choices=["OPEN", "RESOLVED", "SKIPPED", "VOID"])
    list_cmd.add_argument("--category")
    list_cmd.add_argument("--grade", choices=["A", "B", "C", "SKIP"])

    resolve = journal_subparsers.add_parser("resolve", help="Resolve a journal entry.")
    resolve.add_argument("--id", type=int, required=True)
    resolve.add_argument("--resolution-value", type=int, choices=[0, 1], required=True)
    resolve.add_argument("--notes")

    delete = journal_subparsers.add_parser("delete", help="Hard-delete a journal entry.")
    delete.add_argument("--id", type=int, required=True)

    export = journal_subparsers.add_parser("export", help="Export journal entries to CSV.")
    export.add_argument("--output", required=True)
    export.add_argument("--status", choices=["OPEN", "RESOLVED", "SKIPPED", "VOID"])
    export.add_argument("--category")
    export.add_argument("--grade", choices=["A", "B", "C", "SKIP"])

    journal_subparsers.add_parser("summary", help="Summarize journal performance.")

    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = GammaClient()
    clob_client = ClobClient()

    if args.command == "scan":
        markets = client.list_markets(
            active=True,
            closed=False,
            limit=args.limit,
            offset=args.offset,
            order=args.order,
            ascending=False,
        )
        candidates = filter_markets(markets, min_liquidity=args.min_liquidity)
        liquidities = {
            market.id: liquidity
            for market in candidates
            if (liquidity := fetch_binary_liquidity(clob_client, market)) is not None
        }
        if args.mode == "candidates":
            print(candidates_report(candidates, size=args.size, liquidities=liquidities))
        else:
            if args.fair_yes is None:
                parser.error("--fair-yes is required when --mode ev")
            print(
                markets_report(
                    candidates,
                    fair_yes_probability=args.fair_yes,
                    size=args.size,
                    liquidities=liquidities,
                    min_net_edge=args.min_net_edge,
                )
            )
        return 0

    if args.command == "market":
        market = client.get_market(args.market_id)
        liquidity = fetch_binary_liquidity(clob_client, market)
        print(
            market_report(
                market,
                fair_yes_probability=args.fair_yes,
                size=args.size,
                liquidity=liquidity,
                min_net_edge=args.min_net_edge,
            )
        )
        return 0

    if args.command == "slug":
        market = client.get_market_by_slug(args.slug)
        liquidity = fetch_binary_liquidity(clob_client, market)
        print(
            market_report(
                market,
                fair_yes_probability=args.fair_yes,
                size=args.size,
                liquidity=liquidity,
                min_net_edge=args.min_net_edge,
            )
        )
        return 0

    if args.command == "analyze-resolution":
        try:
            market = (
                client.get_market(args.market_id)
                if args.market_id
                else client.get_market_by_slug(args.slug)
            )
            print(resolution_analysis_report(market))
        except (LookupError, MarketStructureError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "strategy":
        if args.strategy_command == "scan":
            markets = client.list_markets(
                active=True,
                closed=False,
                limit=args.limit,
                order="volume",
                ascending=False,
            )
            candidates = filter_markets(markets, min_liquidity=args.min_liquidity)
            liquidities = {
                market.id: liquidity
                for market in candidates
                if (liquidity := fetch_binary_liquidity(clob_client, market)) is not None
            }
            strategy_candidates = scan_strategy_candidates(
                candidates,
                liquidities,
                size=args.size,
                min_grade=args.min_grade,
                max_resolution_risk=args.max_resolution_risk,
                category=args.category,
                include_long_dated=args.include_long_dated,
                include_longshots=args.include_longshots,
            )
            if args.output_json:
                row_count = write_strategy_candidates_json(strategy_candidates, args.output_json)
                print(f"Wrote `{row_count}` strategy candidates to `{args.output_json}`")
            print(strategy_candidates_report(strategy_candidates))
            return 0

    if args.command == "alpha":
        if args.alpha_command == "scan-weather":
            if args.weather_provider == "csv":
                provider = CsvWeatherDataProvider(args.weather_data)
            else:
                provider = OpenMeteoForecastProvider(
                    location_resolver=LocationResolver(args.locations_file),
                    fallback_forecast_std=args.fallback_forecast_std,
                    cache_dir=args.cache_dir,
                    refresh_cache=args.refresh_cache,
                )

            def resolution_lookup(candidate: dict[str, Any]):
                market_id = candidate.get("market_id")
                if market_id is None:
                    return None
                return analyze_market_resolution(client.get_market(str(market_id)))

            calibration_summaries = (
                load_calibration_summaries(args.calibration_json)
                if args.calibration_json
                else None
            )
            result = run_weather_alpha_scan(
                args.strategy_json,
                provider,
                edge_threshold=args.edge_threshold,
                bucket_mode=args.bucket_mode,
                weather_model=args.weather_model,
                student_t_df=args.student_t_df,
                mixture_tail_weight=args.mixture_tail_weight,
                mixture_tail_scale=args.mixture_tail_scale,
                as_of_time=args.as_of_time,
                resolution_lookup=resolution_lookup,
                calibration_summaries=calibration_summaries,
                calibration_group=args.calibration_group,
                use_calibrated_std=args.use_calibrated_std,
                use_calibrated_bias=args.use_calibrated_bias,
                min_calibration_samples=args.min_calibration_samples,
                allow_low_quality_calibration=args.allow_low_quality_calibration,
            )
            if args.output_json:
                row_count = write_weather_alpha_signals_json(result, args.output_json)
                print(f"Wrote `{row_count}` weather alpha signals to `{args.output_json}`")
            print(weather_alpha_report(result))
            return 0
        if args.alpha_command == "diagnose-weather-models":
            try:
                diagnostics = diagnose_weather_models(
                    mean=args.mean,
                    std=args.std,
                    unit=args.unit,
                    bucket_mode=args.bucket_mode,
                    student_t_df=args.student_t_df,
                    mixture_tail_weight=args.mixture_tail_weight,
                    mixture_tail_scale=args.mixture_tail_scale,
                    k_min=args.k_min,
                    k_max=args.k_max,
                )
            except ValueError as exc:
                parser.error(str(exc))
            row_count = write_weather_model_diagnostics_csv(diagnostics, args.output_csv)
            print(f"Wrote `{row_count}` diagnostic rows to `{args.output_csv}`")
            print(weather_model_diagnostics_report(diagnostics))
            return 0

    if args.command == "weather-calibration":
        if args.weather_calibration_command == "fit":
            try:
                group_by = normalize_group_by(args.group_by)
                summaries = fit_weather_calibration(
                    args.input,
                    group_by=group_by,
                    min_samples=args.min_samples,
                    bias_shrinkage_k=args.bias_shrinkage_k,
                )
            except ValueError as exc:
                parser.error(str(exc))
            json_count = write_calibration_json(summaries, args.output_json)
            csv_count = write_calibration_csv(summaries, args.output_csv)
            print(
                _weather_calibration_fit_to_markdown(
                    summaries,
                    json_count=json_count,
                    csv_count=csv_count,
                    output_json=args.output_json,
                    output_csv=args.output_csv,
                )
            )
            return 0

    if args.command == "weather-dataset":
        if args.weather_dataset_command == "build":
            try:
                provider = OpenMeteoHistoricalDatasetProvider(
                    cache_dir=args.cache_dir,
                    refresh_cache=args.refresh_cache,
                    timeout_seconds=args.timeout_seconds,
                    print_request_url=args.print_request_url,
                    proxy=args.proxy,
                    trust_env=args.trust_env,
                )
                summary = build_weather_dataset(
                    locations_file=args.locations_file,
                    output_path=args.output,
                    provider=provider,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    metrics=parse_csv_list(args.metrics),
                    forecast_issue_hours=parse_csv_list(args.forecast_issue_hours, cast=int),
                    horizons=parse_csv_list(args.horizons, cast=int),
                )
                audit = build_provider_semantics_audit(
                    summary=summary,
                    cache_dir=args.cache_dir,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    provider="open_meteo",
                    proxy=args.proxy,
                    trust_env=args.trust_env,
                )
                write_provider_semantics_audit(audit, args.audit_output)
                if audit.smoke_status == "NETWORK_FAILED" and summary.samples_generated == 0:
                    Path(args.output).unlink(missing_ok=True)
            except ValueError as exc:
                parser.error(str(exc))
            print(weather_dataset_summary_to_markdown(summary))
            print()
            print(provider_semantics_audit_to_markdown(audit, args.audit_output))
            return 0
        if args.weather_dataset_command == "debug-provider":
            report = debug_open_meteo_provider(
                location=args.location,
                latitude=args.latitude,
                longitude=args.longitude,
                target_date=args.target_date,
                forecast_issued_at=args.forecast_issued_at,
                metric=args.metric,
                horizon=args.horizon,
                cache_dir=args.cache_dir,
                refresh_cache=args.refresh_cache,
                timeout_seconds=args.timeout_seconds,
                print_request_url=args.print_request_url,
                proxy=args.proxy,
                trust_env=args.trust_env,
            )
            write_provider_debug_report(report, args.output)
            print(provider_debug_report_to_markdown(report, args.output))
            return 0
        if args.weather_dataset_command == "debug-network":
            report = run_network_debug(
                url=args.url,
                timeout_seconds=args.timeout_seconds,
                proxy=args.proxy,
                trust_env=args.trust_env,
                print_env_proxy=args.print_env_proxy,
            )
            write_network_debug_report(report, args.output)
            print(network_debug_report_to_markdown(report))
            return 0
        if args.weather_dataset_command == "manual-template":
            rows = write_manual_forecast_actual_template(
                output_path=args.output,
                locations_file=args.locations_file,
                rows_per_location=args.rows_per_location,
            )
            print(f"Wrote `{rows}` manual template rows to `{args.output}`")
            return 0
        if args.weather_dataset_command == "validate-manual-csv":
            summary = validate_manual_forecast_actual_csv(args.input)
            print(manual_validation_to_markdown(summary))
            return 0
        if args.weather_dataset_command == "data-source-options":
            write_data_source_options_markdown(
                output_path=args.output,
                open_meteo_status=args.open_meteo_status,
                failure_reason=args.failure_reason,
            )
            print(f"Wrote data source options to `{args.output}`")
            return 0

    if args.command == "weather-backtest":
        store = WeatherBacktestStore(settings.weather_backtest_db_path)
        if args.weather_backtest_command == "add-from-signals":
            result = store.add_from_signals(
                args.signals_json,
                entry_size=args.entry_size,
                strict=args.strict,
                include_needs_review=args.include_needs_review,
                allow_unconfirmed_bucket=args.allow_unconfirmed_bucket,
                allow_station_mismatch=args.allow_station_mismatch,
                allow_sample_data=args.allow_sample_data,
                status=args.status,
            )
            print(_weather_backtest_add_result_to_markdown(result.saved, result.skipped))
            return 0
        if args.weather_backtest_command == "list":
            snapshots = store.list_snapshots(
                limit=args.limit,
                status=args.status,
                signal_status=args.signal_status,
                side=args.side,
                location=args.location,
            )
            print(_weather_backtest_snapshots_to_markdown(snapshots))
            return 0
        if args.weather_backtest_command == "resolve":
            snapshot = store.resolve_snapshot(
                snapshot_id=args.id,
                actual_value=args.actual_value,
                resolution_value=args.resolution_value,
                notes=args.notes,
            )
            print(_weather_backtest_snapshot_to_markdown(snapshot))
            return 0
        if args.weather_backtest_command == "summary":
            print(json.dumps(store.summarize(), indent=2, sort_keys=True))
            return 0
        if args.weather_backtest_command == "export":
            row_count = store.export_csv(args.output)
            print(f"Exported `{row_count}` weather backtest snapshots to `{args.output}`")
            return 0

    if args.command == "daily-capture":
        if args.daily_capture_command == "weather":
            try:
                config = DailyWeatherCaptureConfig(
                    limit=args.limit,
                    min_liquidity=args.min_liquidity,
                    size=args.size,
                    weather_provider=args.weather_provider,
                    weather_data=args.weather_data,
                    calibration_json=args.calibration_json,
                    weather_model=args.weather_model,
                    bucket_mode=args.bucket_mode,
                    proxy=args.proxy,
                    trust_env=args.trust_env,
                    output_dir=args.output_dir,
                    snapshot_db=args.snapshot_db,
                    backtest_db=args.backtest_db,
                    entry_size=args.entry_size,
                    strict=args.strict,
                    include_needs_review=args.include_needs_review,
                    dry_run=args.dry_run,
                    sizes=parse_csv_list(args.sizes, cast=float),
                    forecast_time_tolerance_seconds=args.forecast_time_tolerance_seconds,
                )
            except ValueError as exc:
                parser.error(str(exc))
            summary = run_daily_weather_capture(config)
            print(daily_capture_summary_to_markdown(summary))
            return summary.exit_code

    if args.command == "journal":
        journal = ResearchJournal(settings.journal_db_path)
        if args.journal_command == "add":
            payload = _journal_add_payload_from_args(args, parser)
            warnings = _expected_value_warnings(journal, payload)
            entry = journal.create_entry(
                market_id=payload["market_id"],
                slug=payload.get("slug"),
                question=payload["question"],
                category=payload.get("category"),
                end_date=payload.get("end_date"),
                candidate_score=payload.get("candidate_score"),
                candidate_grade=payload.get("candidate_grade"),
                side=payload["side"],
                fair_yes_probability=payload.get("fair_yes_probability"),
                probability_source=payload.get("probability_source") or "unknown",
                entry_price=payload.get("entry_price"),
                entry_size=payload.get("entry_size"),
                fee_per_share=payload.get("fee_per_share"),
                expected_value_per_share=payload.get("expected_value_per_share"),
                expected_profit=payload.get("expected_profit"),
                rationale=payload.get("rationale"),
                status=payload.get("status"),
                notes=payload.get("notes"),
            )
            for warning in warnings:
                print(f"Warning: {warning}")
            print(_entry_to_markdown(entry))
            return 0
        if args.journal_command == "list":
            entries = journal.list_entries(
                limit=args.limit,
                status=args.status,
                category=args.category,
                grade=args.grade,
            )
            print(_entries_to_markdown(entries))
            return 0
        if args.journal_command == "resolve":
            entry = journal.update_resolution(
                entry_id=args.id,
                resolution_value=args.resolution_value,
                notes=args.notes,
            )
            print(_entry_to_markdown(entry))
            return 0
        if args.journal_command == "delete":
            try:
                journal.delete_entry(args.id)
            except LookupError as exc:
                parser.error(str(exc))
            print(f"Deleted journal entry `{args.id}`")
            return 0
        if args.journal_command == "export":
            row_count = journal.export_csv(
                args.output,
                status=args.status,
                category=args.category,
                grade=args.grade,
            )
            print(f"Exported `{row_count}` journal entries to `{args.output}`")
            return 0
        if args.journal_command == "summary":
            print(json.dumps(journal.summarize_performance(), indent=2, sort_keys=True))
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _journal_add_payload_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.from_json_file:
        path = Path(args.from_json_file)
        try:
            with path.open(encoding="utf-8") as file:
                loaded = json.load(file)
        except OSError as exc:
            parser.error(f"could not read --from-json-file: {exc}")
        except json.JSONDecodeError as exc:
            parser.error(f"invalid JSON in --from-json-file: {exc}")
        if not isinstance(loaded, dict):
            parser.error("--from-json-file must contain a JSON object")
        payload.update(loaded)

    cli_values = {
        "market_id": args.market_id,
        "slug": args.slug,
        "question": args.question,
        "category": args.category,
        "end_date": args.end_date,
        "candidate_score": args.candidate_score,
        "candidate_grade": args.candidate_grade,
        "side": args.side,
        "fair_yes_probability": args.fair_yes,
        "probability_source": args.probability_source,
        "entry_price": args.entry_price,
        "entry_size": args.entry_size,
        "fee_per_share": args.fee_per_share,
        "expected_value_per_share": args.expected_value_per_share,
        "expected_profit": args.expected_profit,
        "rationale": args.rationale,
        "status": args.status,
        "notes": args.notes,
    }
    for key, value in cli_values.items():
        if value is not None:
            payload[key] = value

    missing = [key for key in ("market_id", "question", "side") if not payload.get(key)]
    if missing:
        parser.error(f"journal add requires {', '.join(missing)}")
    side = str(payload.get("side", "NONE")).upper()
    if side in {"YES", "NO"} and payload.get("fair_yes_probability") is None:
        parser.error(
            "journal add requires --fair-yes when side is YES or NO; "
            "strategy drafts intentionally leave fair_yes_probability null"
        )
    return payload


def _expected_value_warnings(journal: ResearchJournal, payload: dict[str, Any]) -> list[str]:
    side = str(payload.get("side", "NONE")).upper()
    auto_ev, auto_profit = journal._expected_values(
        side=side,
        fair_yes_probability=payload.get("fair_yes_probability"),
        entry_price=payload.get("entry_price"),
        entry_size=payload.get("entry_size"),
        fee_per_share=payload.get("fee_per_share"),
    )
    warnings: list[str] = []
    explicit_ev = payload.get("expected_value_per_share")
    explicit_profit = payload.get("expected_profit")
    if explicit_ev is not None and auto_ev is not None and abs(explicit_ev - auto_ev) > 1e-6:
        warnings.append(
            "expected_value_per_share differs from auto calculation "
            f"({explicit_ev:.6g} vs {auto_ev:.6g})"
        )
    if (
        explicit_profit is not None
        and auto_profit is not None
        and abs(explicit_profit - auto_profit) > 1e-6
    ):
        warnings.append(
            "expected_profit differs from auto calculation "
            f"({explicit_profit:.6g} vs {auto_profit:.6g})"
        )
    return warnings


def _entry_to_markdown(entry: JournalEntry) -> str:
    return "\n".join(
        [
            "# Journal Entry",
            "",
            f"- id: `{entry.id}`",
            f"- market_id: `{entry.market_id}`",
            f"- slug: `{entry.slug or 'n/a'}`",
            f"- question: `{entry.question}`",
            f"- category: `{entry.category or 'n/a'}`",
            f"- candidate_score: `{_fmt_optional(entry.candidate_score)}`",
            f"- candidate_grade: `{entry.candidate_grade or 'n/a'}`",
            f"- side: `{entry.side}`",
            f"- fair_yes_probability: `{_fmt_optional(entry.fair_yes_probability)}`",
            f"- probability_source: `{entry.probability_source}`",
            f"- entry_price: `{_fmt_optional(entry.entry_price)}`",
            f"- entry_size: `{_fmt_optional(entry.entry_size)}`",
            f"- fee_per_share: `{_fmt_optional(entry.fee_per_share)}`",
            f"- expected_value_per_share: `{_fmt_optional(entry.expected_value_per_share)}`",
            f"- expected_profit: `{_fmt_optional(entry.expected_profit)}`",
            f"- status: `{entry.status}`",
            f"- resolution_value: `{entry.resolution_value if entry.resolution_value is not None else 'n/a'}`",
            f"- realized_pnl: `{_fmt_optional(entry.realized_pnl)}`",
            f"- brier_score: `{_fmt_optional(entry.brier_score)}`",
        ]
    )


def _entries_to_markdown(entries: list[JournalEntry]) -> str:
    lines = [
        "# Journal Entries",
        "",
        "| ID | Created | Status | Market | Side | Fair YES | Entry | Size | PnL | Brier |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        lines.append(
            f"| {entry.id} | {entry.created_at} | {entry.status} | {entry.market_id} | "
            f"{entry.side} | {_fmt_optional(entry.fair_yes_probability)} | "
            f"{_fmt_optional(entry.entry_price)} | {_fmt_optional(entry.entry_size)} | "
            f"{_fmt_optional(entry.realized_pnl)} | {_fmt_optional(entry.brier_score)} |"
        )
    return "\n".join(lines)


def _weather_backtest_add_result_to_markdown(
    saved: list[WeatherBacktestSnapshot],
    skipped: list[str],
) -> str:
    lines = [
        "# Weather Backtest Add From Signals",
        "",
        "- Dataset type: `forward_replay_paper_dataset`",
        "- This is not a historical Polymarket backtest without historical orderbook snapshots.",
        f"- saved_snapshots: `{len(saved)}`",
        f"- skipped_signals: `{len(skipped)}`",
    ]
    if skipped:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- `{item}`" for item in skipped)
    if saved:
        lines.extend(["", "## Saved", ""])
        for snapshot in saved:
            lines.append(
                f"- id `{snapshot.id}` market `{snapshot.market_id}` side "
                f"`{snapshot.suggested_paper_side}` status `{snapshot.status}` "
                f"notes `{snapshot.notes or 'n/a'}`"
            )
    return "\n".join(lines)


def _weather_backtest_snapshots_to_markdown(
    snapshots: list[WeatherBacktestSnapshot],
) -> str:
    lines = [
        "# Weather Backtest Snapshots",
        "",
        "- Dataset type: `forward_replay_paper_dataset`",
        "- This is not a historical Polymarket backtest without historical orderbook snapshots.",
        "",
        "| ID | Status | Signal | Side | Market | Location | p_yes | EV/share | PnL | Brier |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for snapshot in snapshots:
        lines.append(
            f"| {snapshot.id} | {snapshot.status} | {snapshot.signal_status or 'n/a'} | "
            f"{snapshot.suggested_paper_side} | {snapshot.market_id} | "
            f"{snapshot.location_name or 'n/a'} | {_fmt_optional(snapshot.model_p_yes)} | "
            f"{_fmt_optional(snapshot.expected_value_per_share)} | "
            f"{_fmt_optional(snapshot.realized_pnl)} | {_fmt_optional(snapshot.brier_score)} |"
        )
    return "\n".join(lines)


def _weather_backtest_snapshot_to_markdown(snapshot: WeatherBacktestSnapshot) -> str:
    return "\n".join(
        [
            "# Weather Backtest Snapshot",
            "",
            f"- id: `{snapshot.id}`",
            f"- market_id: `{snapshot.market_id}`",
            f"- question: `{snapshot.question}`",
            f"- location: `{snapshot.location_name or 'n/a'}`",
            f"- side: `{snapshot.suggested_paper_side}`",
            f"- model_p_yes: `{_fmt_optional(snapshot.model_p_yes)}`",
            f"- entry_price: `{_fmt_optional(snapshot.entry_price)}`",
            f"- entry_size: `{_fmt_optional(snapshot.entry_size)}`",
            f"- status: `{snapshot.status}`",
            f"- actual_value: `{_fmt_optional(snapshot.actual_value)}`",
            f"- resolution_value: `{snapshot.resolution_value if snapshot.resolution_value is not None else 'n/a'}`",
            f"- realized_pnl: `{_fmt_optional(snapshot.realized_pnl)}`",
            f"- brier_score: `{_fmt_optional(snapshot.brier_score)}`",
            f"- notes: `{snapshot.notes or 'n/a'}`",
        ]
    )


def _weather_calibration_fit_to_markdown(
    summaries: list[WeatherCalibrationSummary],
    *,
    json_count: int,
    csv_count: int,
    output_json: str,
    output_csv: str,
) -> str:
    lines = [
        "# Weather Calibration Fit",
        "",
        f"- summaries: `{len(summaries)}`",
        f"- wrote_json: `{json_count}` rows to `{output_json}`",
        f"- wrote_csv: `{csv_count}` rows to `{output_csv}`",
        "",
        "| Group | quality | n | min | bias_raw | bias_shrunk | std_used | mae | rmse |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries[:20]:
        lines.append(
            f"| {summary.group_key} | {summary.calibration_quality} | {summary.n} | "
            f"{summary.min_samples_required} | {_fmt_optional(summary.bias_raw)} | "
            f"{_fmt_optional(summary.bias_shrunk)} | {_fmt_optional(summary.std_error_used)} | "
            f"{summary.mae:.6g} | {summary.rmse:.6g} |"
        )
    return "\n".join(lines)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
