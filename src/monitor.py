"""
Monitoring module - drift detection.

Responsibility:
- Compare training data distribution with new (production) data
- Detect drift using statistical tests (KS-test)
- Warn if drift exceeds threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from src.data import get_feature_columns, load_raw_data


DEFAULT_REFERENCE_PATH = Path("data/raw/sensor_data.csv")
DEFAULT_CURRENT_PATH = Path("data/raw/sensor_data.csv")
DEFAULT_REPORT_PATH = Path("models/drift_report.json")
DRIFT_THRESHOLD = 0.05


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """Drift detection using Kolmogorov-Smirnov test."""
    results = {}
    n_drifted = 0

    for col in feature_cols:
        ref_values = reference_df[col].dropna().values
        cur_values = current_df[col].dropna().values

        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        is_drifted = p_value < threshold

        results[col] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "is_drifted": bool(is_drifted),
        }

        if is_drifted:
            n_drifted += 1

    return {
        "threshold": threshold,
        "n_features": len(feature_cols),
        "n_drifted": n_drifted,
        "drift_rate": n_drifted / len(feature_cols),
        "features": results,
    }


def print_report(report: dict) -> None:
    """Human-readable report."""
    print("=" * 60)
    print("DRIFT DETECTION REPORT")
    print("=" * 60)
    print("Threshold (p-value): {}".format(report["threshold"]))
    print("Features tested:     {}".format(report["n_features"]))
    print("Features drifted:    {}".format(report["n_drifted"]))
    print("Drift rate:          {:.1%}".format(report["drift_rate"]))
    print("-" * 60)

    sorted_features = sorted(
        report["features"].items(),
        key=lambda x: x[1]["ks_statistic"],
        reverse=True,
    )

    print("{:<10} {:>10} {:>12} {:>10}".format("Feature", "KS stat", "p-value", "Drifted"))
    print("-" * 60)
    for name, data in sorted_features:
        flag = "YES" if data["is_drifted"] else "no"
        print(
            "{:<10} {:>10.4f} {:>12.6f} {:>10}".format(
                name, data["ks_statistic"], data["p_value"], flag
            )
        )
    print("=" * 60)

    if report["n_drifted"] > 0:
        print("\nWARNING: {} feature(s) show significant drift!".format(report["n_drifted"]))
        print("Consider retraining the model.")
    else:
        print("\nOK: No significant drift detected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift detection")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    args = parser.parse_args()

    print("Reference data: {}".format(args.reference))
    print("Current data:   {}".format(args.current))
    print()

    reference_df = load_raw_data(args.reference)
    current_df = load_raw_data(args.current)

    feature_cols = get_feature_columns(reference_df)

    report = detect_drift(
        reference_df, current_df, feature_cols, threshold=args.threshold
    )

    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved: {}".format(args.output))


if __name__ == "__main__":
    main()