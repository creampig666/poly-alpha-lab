"""Diagnostics for comparing weather bucket probability model shapes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from poly_alpha_lab.weather_data import WeatherForecast
from poly_alpha_lab.weather_probability_model import (
    BucketMode,
    TemperatureUnit,
    estimate_temperature_threshold_probability,
)

DiagnosticUnit = Literal["C", "F"]

DIAGNOSTIC_FIELDS = [
    "K",
    "normal_bucket_probability",
    "student_t_bucket_probability",
    "normal_mixture_bucket_probability",
    "student_t_minus_normal",
    "mixture_minus_normal",
]


class WeatherModelDiagnosticRow(BaseModel):
    K: int
    normal_bucket_probability: float
    student_t_bucket_probability: float
    normal_mixture_bucket_probability: float
    student_t_minus_normal: float
    mixture_minus_normal: float


class WeatherModelDiagnostics(BaseModel):
    mean: float
    std: float
    unit: DiagnosticUnit
    bucket_mode: BucketMode
    student_t_df: float
    mixture_tail_weight: float
    mixture_tail_scale: float
    rows: list[WeatherModelDiagnosticRow]

    @property
    def center_k(self) -> int:
        return int(round(self.mean))

    @property
    def center_row(self) -> WeatherModelDiagnosticRow | None:
        return self.row_for_k(self.center_k)

    @property
    def left_tail_row(self) -> WeatherModelDiagnosticRow | None:
        return self.rows[0] if self.rows else None

    @property
    def right_tail_row(self) -> WeatherModelDiagnosticRow | None:
        return self.rows[-1] if self.rows else None

    def row_for_k(self, k: int) -> WeatherModelDiagnosticRow | None:
        for row in self.rows:
            if row.K == k:
                return row
        return None

    def student_t_greater_than_normal(self) -> list[int]:
        return [row.K for row in self.rows if row.student_t_minus_normal > 0]

    def mixture_greater_than_normal(self) -> list[int]:
        return [row.K for row in self.rows if row.mixture_minus_normal > 0]


def diagnose_weather_models(
    *,
    mean: float,
    std: float,
    unit: DiagnosticUnit = "C",
    bucket_mode: BucketMode = "rounded",
    student_t_df: float = 5,
    mixture_tail_weight: float = 0.10,
    mixture_tail_scale: float = 2.5,
    k_min: int = 18,
    k_max: int = 30,
) -> WeatherModelDiagnostics:
    """Compare exact bucket probabilities across normal, student-t, and mixture models."""

    if k_min > k_max:
        raise ValueError("k_min must be less than or equal to k_max")
    forecast = WeatherForecast(
        date="2026-01-01",
        location="diagnostic",
        metric="high_temperature",
        forecast_mean=mean,
        forecast_std=std,
        unit=unit,
    )
    rows: list[WeatherModelDiagnosticRow] = []
    for k in range(k_min, k_max + 1):
        normal = estimate_temperature_threshold_probability(
            forecast=forecast,
            threshold=k,
            comparator="exact_bucket",
            threshold_unit=unit,
            bucket_mode=bucket_mode,
            weather_model="normal",
        ).model_p_yes
        student_t = estimate_temperature_threshold_probability(
            forecast=forecast,
            threshold=k,
            comparator="exact_bucket",
            threshold_unit=unit,
            bucket_mode=bucket_mode,
            weather_model="student_t",
            student_t_df=student_t_df,
        ).model_p_yes
        mixture = estimate_temperature_threshold_probability(
            forecast=forecast,
            threshold=k,
            comparator="exact_bucket",
            threshold_unit=unit,
            bucket_mode=bucket_mode,
            weather_model="normal_mixture",
            mixture_tail_weight=mixture_tail_weight,
            mixture_tail_scale=mixture_tail_scale,
        ).model_p_yes
        rows.append(
            WeatherModelDiagnosticRow(
                K=k,
                normal_bucket_probability=normal,
                student_t_bucket_probability=student_t,
                normal_mixture_bucket_probability=mixture,
                student_t_minus_normal=student_t - normal,
                mixture_minus_normal=mixture - normal,
            )
        )
    return WeatherModelDiagnostics(
        mean=mean,
        std=std,
        unit=unit,
        bucket_mode=bucket_mode,
        student_t_df=student_t_df,
        mixture_tail_weight=mixture_tail_weight,
        mixture_tail_scale=mixture_tail_scale,
        rows=rows,
    )


def write_weather_model_diagnostics_csv(
    diagnostics: WeatherModelDiagnostics,
    output_path: str | Path,
) -> int:
    """Write diagnostics rows to CSV and return row count."""

    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in diagnostics.rows:
            writer.writerow(row.model_dump())
    return len(diagnostics.rows)


def weather_model_diagnostics_report(diagnostics: WeatherModelDiagnostics) -> str:
    """Render model diagnostics as Markdown."""

    lines = [
        "# Weather Model Diagnostics",
        "",
        f"- mean: `{diagnostics.mean:g}{diagnostics.unit}`",
        f"- std: `{diagnostics.std:g}{diagnostics.unit}`",
        f"- bucket_mode: `{diagnostics.bucket_mode}`",
        f"- student_t_df: `{diagnostics.student_t_df:g}`",
        f"- mixture_tail_weight: `{diagnostics.mixture_tail_weight:g}`",
        f"- mixture_tail_scale: `{diagnostics.mixture_tail_scale:g}`",
        "",
        "| K | Normal | Student-t | Normal Mixture | Student-t - Normal | Mixture - Normal |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diagnostics.rows:
        lines.append(
            f"| {row.K} | {row.normal_bucket_probability:.6f} | "
            f"{row.student_t_bucket_probability:.6f} | "
            f"{row.normal_mixture_bucket_probability:.6f} | "
            f"{row.student_t_minus_normal:.6f} | {row.mixture_minus_normal:.6f} |"
        )
    lines.extend(["", "## Summary", ""])
    if diagnostics.center_row:
        lines.append(
            f"- Center bucket K={diagnostics.center_row.K}: "
            f"normal `{diagnostics.center_row.normal_bucket_probability:.6f}`, "
            f"student_t `{diagnostics.center_row.student_t_bucket_probability:.6f}`, "
            f"normal_mixture `{diagnostics.center_row.normal_mixture_bucket_probability:.6f}`"
        )
    if diagnostics.left_tail_row:
        lines.append(
            f"- Left tail bucket K={diagnostics.left_tail_row.K}: "
            f"normal `{diagnostics.left_tail_row.normal_bucket_probability:.6f}`, "
            f"student_t `{diagnostics.left_tail_row.student_t_bucket_probability:.6f}`, "
            f"normal_mixture `{diagnostics.left_tail_row.normal_mixture_bucket_probability:.6f}`"
        )
    if diagnostics.right_tail_row:
        lines.append(
            f"- Right tail bucket K={diagnostics.right_tail_row.K}: "
            f"normal `{diagnostics.right_tail_row.normal_bucket_probability:.6f}`, "
            f"student_t `{diagnostics.right_tail_row.student_t_bucket_probability:.6f}`, "
            f"normal_mixture `{diagnostics.right_tail_row.normal_mixture_bucket_probability:.6f}`"
        )
    lines.append(
        f"- Buckets where student_t > normal: `{_fmt_ints(diagnostics.student_t_greater_than_normal())}`"
    )
    lines.append(
        f"- Buckets where normal_mixture > normal: `{_fmt_ints(diagnostics.mixture_greater_than_normal())}`"
    )
    return "\n".join(lines) + "\n"


def _fmt_ints(values: list[int]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"
