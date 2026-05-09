# poly-alpha-lab

`poly-alpha-lab` is a Python 3.11+ read-only research toolkit for Polymarket markets.
It uses `httpx` against the public Polymarket Gamma and CLOB APIs and does not
require a wallet, private key, or trading credentials.

Official API references used for this version:

- Gamma API base URL: `https://gamma-api.polymarket.com`
- Market list: `GET /markets`
- Market by ID: `GET /markets/{id}`
- CLOB API base URL: `https://clob.polymarket.com`
- Order book by token ID: `GET /book?token_id=...`

## Install

```powershell
cd C:\Users\28983\dev\poly-alpha-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

## CLI

Scan active markets for human-research candidates:

```powershell
poly-alpha-lab scan --mode candidates --limit 20 --min-liquidity 1000 --size 10
```

Run conditional EV mode with a user-provided fair YES probability:

```powershell
poly-alpha-lab scan --mode ev --limit 10 --min-liquidity 1000 --fair-yes 0.55 --size 100
```

Fetch one market by Gamma market ID:

```powershell
poly-alpha-lab market 12345 --fair-yes 0.55 --size 100
```

Generate paper strategy candidates with breakeven probabilities:

```powershell
poly-alpha-lab strategy scan --limit 100 --min-liquidity 1000 --size 10 --output-json strategy_candidates_daily.json
```

Run the weather temperature paper alpha module from strategy JSON and a local
forecast CSV. It supports threshold markets and exact integer temperature bucket
markets:

```powershell
poly-alpha-lab alpha scan-weather --strategy-json strategy_candidates_daily.json --weather-data data/weather/weather_forecasts.csv --bucket-mode rounded --output-json weather_alpha_signals.json
```

Build a weather forward replay dataset from saved weather alpha signals:

```powershell
poly-alpha-lab weather-backtest add-from-signals --signals-json weather_alpha_signals.json --entry-size 10 --strict
poly-alpha-lab weather-backtest list --limit 20
poly-alpha-lab weather-backtest summary
```

Weather backtest storage is a forward replay / paper dataset. Without historical
Polymarket order book or price snapshots, it is not a historical Polymarket
backtest. Strict replay rejects sample/manual forecast data and enforces
`forecast_issued_at <= as_of_time`.

Capture a daily forward replay slice for future paper evaluation:

```powershell
poly-alpha-lab daily-capture weather --limit 100 --min-liquidity 1000 --size 10 --weather-provider open-meteo --calibration-json data/weather/calibration_summary_real_pilot_30d.json --weather-model normal --bucket-mode rounded --proxy http://127.0.0.1:7897 --no-trust-env --output-dir data/daily --snapshot-db data/polymarket_snapshots.sqlite --backtest-db data/weather_backtest.sqlite --entry-size 10 --strict --forecast-time-tolerance-seconds 120
```

`daily-capture weather` is a forward replay capture job. It saves strategy
candidates, weather alpha signals, public CLOB order-book snapshots, and optional
paper replay rows. It is not a historical Polymarket backtest, because it only
captures order books from the moment the job runs. It never places orders, never
authenticates, never reads wallet data, and never uses private keys.
Live capture records both pipeline `captured_at` and alpha `alpha_as_of_time`;
current forecast timestamps within `--forecast-time-tolerance-seconds` are
marked as live capture timing tolerance rather than future data.

For Windows Task Scheduler, run `scripts/run_daily_weather_capture.ps1` once per
day or hour from the project root. Adjust the proxy and calibration path for your
machine.

The default `scan` mode is `candidates`, which reports market quality scores and
candidate grades only. It does not output BUY/SELL direction. EV mode is conditional
on a user-provided `fair_yes_probability`; trading judgment uses full-size executable
CLOB average buy price only. Gamma `outcomePrices`, midpoint, and last trade are
displayed as indicative context only.

## Tests

```powershell
pytest
```

## Scope

Version 1.3.2 is intentionally read-only and paper-only:

- No private key handling
- No order signing
- No order placement or cancellation
- No authenticated CLOB endpoints
- No automatic journal writes
- No external LLM probability model

## Journal

The local journal records manual research and paper trades only:

```powershell
poly-alpha-lab journal add --market-id test-1 --slug test-market --question "Test market?" --category economics --candidate-score 80 --candidate-grade A --side YES --fair-yes 0.6 --probability-source manual --entry-price 0.5 --entry-size 10 --fee-per-share 0.001 --rationale "test entry"
poly-alpha-lab journal list --limit 5
poly-alpha-lab journal resolve --id 1 --resolution-value 1
poly-alpha-lab journal summary
```
