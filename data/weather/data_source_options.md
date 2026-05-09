# Weather Data Source Options

## Open-Meteo Current Failure

- status: `NETWORK_FAILED`
- observed_failure_reason: `ConnectError WinError 10061 connection refused`
- current stance: do not treat mock/sample cache as real data.

## Options

| Source | Historical forecast | Actuals | Station-level | API key | Calibration fit | Strict backtest fit | Main risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Open-Meteo Historical Forecast | Yes, but as-of/run semantics must be audited | Indirect/model archive | Grid/city, not true station | No | Potentially useful after semantics validation | Needs explicit as-of confidence | Forecast run time may not match requested `forecast_issued_at` |
| Open-Meteo Historical Weather only | No | Yes/model archive | Grid/city | No | No, actuals only | No for forecast calibration | Actual archive cannot be used as forecast |
| NOAA / NCEI actual observations | No forecast in this path | Yes | Often station-level | Sometimes/no depending endpoint | Actual side only | Useful for resolution/actual validation | Station mapping and access complexity |
| Meteostat | No forecast baseline | Yes | Station-level possible | Usually no/free tiers vary | Actual side only | Useful for actual validation | Coverage and station continuity |
| Visual Crossing historical forecast | Potentially yes | Yes | Location/grid, depends plan | Usually yes | Possible | Possible if API exposes forecast issue/run timing | API key/cost and semantics need audit |
| Self-collected live forecast snapshots | Yes from collection start onward | Needs paired actual source | Whatever provider supplies | Depends provider | Best for forward calibration | Strong for strict replay after enough samples | Requires waiting to accumulate data |

## Recommendation

- If Open-Meteo failed because of network/timeout/proxy, first fix network or run in a stable environment.
- If Open-Meteo does not expose the required historical forecast as-of semantics, use live forecast snapshot accumulation or a provider with explicit historical forecast run times.
- If a source only provides actual observations, use it for resolution/actual_value only; never use it as forecast_mean.

## Minimal Next Patch

1. Keep `debug-provider` as the first smoke gate.
2. Add a provider only after a single-market debug proves forecast and actual fields are semantically separated.
3. For strict backtest, require forecast snapshots with `forecast_issued_at <= as_of_time` and verified source metadata.
