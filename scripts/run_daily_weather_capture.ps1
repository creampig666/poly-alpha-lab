Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path (Resolve-Path "$PSScriptRoot\..")

poly-alpha-lab daily-capture weather `
  --limit 100 `
  --min-liquidity 1000 `
  --size 10 `
  --weather-provider open-meteo `
  --calibration-json data/weather/calibration_summary_real_pilot_30d.json `
  --weather-model normal `
  --bucket-mode rounded `
  --proxy http://127.0.0.1:7897 `
  --no-trust-env `
  --output-dir data/daily `
  --snapshot-db data/polymarket_snapshots.sqlite `
  --backtest-db data/weather_backtest.sqlite `
  --entry-size 10 `
  --strict `
  --forecast-time-tolerance-seconds 120
