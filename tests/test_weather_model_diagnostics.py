import csv

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.weather_model_diagnostics import (
    DIAGNOSTIC_FIELDS,
    diagnose_weather_models,
    write_weather_model_diagnostics_csv,
)


def test_diagnostics_center_bucket_student_t_can_exceed_normal() -> None:
    diagnostics = diagnose_weather_models(mean=24, std=1, k_min=18, k_max=30)
    center = diagnostics.center_row

    assert center is not None
    assert center.K == 24
    assert center.student_t_bucket_probability > center.normal_bucket_probability


def test_diagnostics_far_tail_student_t_exceeds_normal() -> None:
    diagnostics = diagnose_weather_models(mean=24, std=1, k_min=18, k_max=30)
    right_tail = diagnostics.right_tail_row

    assert right_tail is not None
    assert right_tail.K == 30
    assert right_tail.student_t_bucket_probability > right_tail.normal_bucket_probability


def test_diagnostics_far_tail_mixture_exceeds_normal() -> None:
    diagnostics = diagnose_weather_models(mean=24, std=1, k_min=18, k_max=30)
    right_tail = diagnostics.right_tail_row

    assert right_tail is not None
    assert right_tail.normal_mixture_bucket_probability > right_tail.normal_bucket_probability


def test_diagnostics_csv_fields_complete(tmp_path) -> None:
    diagnostics = diagnose_weather_models(mean=24, std=1, k_min=23, k_max=25)
    output = tmp_path / "diagnostics.csv"

    row_count = write_weather_model_diagnostics_csv(diagnostics, output)

    assert row_count == 3
    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == DIAGNOSTIC_FIELDS
    assert rows[0]["K"] == "23"


def test_diagnostic_cli_runs(tmp_path, capsys) -> None:
    output = tmp_path / "weather_model_diagnostics.csv"

    exit_code = cli.run(
        [
            "alpha",
            "diagnose-weather-models",
            "--mean",
            "24",
            "--std",
            "1",
            "--bucket-mode",
            "rounded",
            "--student-t-df",
            "5",
            "--mixture-tail-weight",
            "0.10",
            "--mixture-tail-scale",
            "2.5",
            "--k-min",
            "18",
            "--k-max",
            "30",
            "--output-csv",
            str(output),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert output.exists()
    assert "Center bucket K=24" in captured
    assert "Buckets where student_t > normal" in captured
