from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "timestamp",
    "yaw",
    "pitch",
    "roll",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
]

SENSOR_COLUMNS = [c for c in NUMERIC_COLUMNS if c != "timestamp"]
ANGLE_COLUMNS = {"yaw", "pitch", "roll"}

# Less-sensitive outlier settings to reduce false positives.
IQR_MULTIPLIER = 3.0
Z_SCORE_THRESHOLD = 4.0
ROBUST_Z_THRESHOLD = 4.5
JUMP_MAD_MULTIPLIER = 60.0
JUMP_FALLBACK_QUANTILE = 0.99999
MIN_DEGREE_JUMP = 2.0
ENABLE_STATISTICAL_OUTLIER_CHECK = False


def robust_z_score(series: pd.Series) -> pd.Series:
    """Return robust z-score based on median absolute deviation."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median(skipna=True)
    abs_dev = (values - median).abs()
    mad = abs_dev.median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return 0.6745 * (values - median) / mad


def standard_z_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0, skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean(skipna=True)) / std


def summarize_column(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    abs_dev = (values - values.median(skipna=True)).abs()
    return {
        "count": float(values.count()),
        "missing": float(values.isna().sum()),
        "mean": float(values.mean(skipna=True)),
        "std": float(values.std(skipna=True)),
        "min": float(values.min(skipna=True)),
        "q01": float(values.quantile(0.01)),
        "median": float(values.median(skipna=True)),
        "q99": float(values.quantile(0.99)),
        "max": float(values.max(skipna=True)),
        "mad": float(abs_dev.median(skipna=True)),
    }


def detect_time_issues(df: pd.DataFrame) -> List[dict]:
    issues: List[dict] = []
    ts = pd.to_numeric(df["timestamp"], errors="coerce")

    dup_mask = ts.duplicated(keep=False)
    for idx in df.index[dup_mask]:
        issues.append(
            {
                "index": int(idx),
                "timestamp": float(ts.loc[idx]),
                "column": "timestamp",
                "value": float(ts.loc[idx]),
                "kind": "duplicate_timestamp",
                "severity": 3.0,
                "details": "Exact duplicate timestamp found.",
            }
        )

    dt = ts.diff()
    non_monotonic_mask = dt <= 0
    non_monotonic_mask.iloc[0] = False
    for idx in df.index[non_monotonic_mask]:
        issues.append(
            {
                "index": int(idx),
                "timestamp": float(ts.loc[idx]),
                "column": "timestamp",
                "value": float(ts.loc[idx]),
                "kind": "non_monotonic_time",
                "severity": 4.0,
                "details": f"Timestamp decreased or repeated (dt={float(dt.loc[idx]):.6f}).",
            }
        )

    positive_dt = dt[dt > 0]
    if not positive_dt.empty:
        med_dt = positive_dt.median()
        mad_dt = (positive_dt - med_dt).abs().median()
        if pd.isna(mad_dt) or mad_dt == 0:
            threshold = positive_dt.quantile(0.99)
        else:
            threshold = med_dt + 8.0 * mad_dt
        gap_mask = dt > threshold
        for idx in df.index[gap_mask]:
            issues.append(
                {
                    "index": int(idx),
                    "timestamp": float(ts.loc[idx]),
                    "column": "timestamp",
                    "value": float(ts.loc[idx]),
                    "kind": "large_time_gap",
                    "severity": float(dt.loc[idx] / threshold) if threshold > 0 else 1.0,
                    "details": f"Large gap between samples (dt={float(dt.loc[idx]):.6f}, threshold={float(threshold):.6f}).",
                }
            )

    return issues


def detect_signal_issues(df: pd.DataFrame) -> List[dict]:
    issues: List[dict] = []
    ts = pd.to_numeric(df["timestamp"], errors="coerce")

    for col in SENSOR_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")

        if ENABLE_STATISTICAL_OUTLIER_CHECK:
            z = standard_z_score(values)
            rz = robust_z_score(values)
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                iqr_outlier = pd.Series(False, index=values.index)
            else:
                low = q1 - IQR_MULTIPLIER * iqr
                high = q3 + IQR_MULTIPLIER * iqr
                iqr_outlier = (values < low) | (values > high)

            z_outlier = z.abs() > Z_SCORE_THRESHOLD
            rz_outlier = rz.abs() > ROBUST_Z_THRESHOLD
            outlier_mask = iqr_outlier | z_outlier | rz_outlier

            for idx in df.index[outlier_mask.fillna(False)]:
                z_val = z.loc[idx]
                rz_val = rz.loc[idx]
                score = max(
                    float(abs(z_val) / Z_SCORE_THRESHOLD) if not pd.isna(z_val) else 0.0,
                    float(abs(rz_val) / ROBUST_Z_THRESHOLD) if not pd.isna(rz_val) else 0.0,
                )
                tests = []
                if bool(iqr_outlier.loc[idx]):
                    tests.append("IQR")
                if bool(z_outlier.loc[idx]):
                    tests.append("z-score")
                if bool(rz_outlier.loc[idx]):
                    tests.append("robust-z")

                issues.append(
                    {
                        "index": int(idx),
                        "timestamp": float(ts.loc[idx]) if not pd.isna(ts.loc[idx]) else np.nan,
                        "column": col,
                        "value": float(values.loc[idx]) if not pd.isna(values.loc[idx]) else np.nan,
                        "kind": "statistical_outlier",
                        "severity": score,
                        "details": f"Flagged by {', '.join(tests)}.",
                    }
                )

        abs_diff = values.diff().abs()
        diff_med = abs_diff.median(skipna=True)
        diff_mad = (abs_diff - diff_med).abs().median(skipna=True)
        if pd.notna(diff_med):
            if pd.isna(diff_mad) or diff_mad == 0:
                jump_threshold = abs_diff.quantile(JUMP_FALLBACK_QUANTILE)
            else:
                jump_threshold = diff_med + JUMP_MAD_MULTIPLIER * diff_mad

            if pd.notna(jump_threshold) and jump_threshold > 0:
                effective_jump_threshold = jump_threshold
                if col in ANGLE_COLUMNS:
                    effective_jump_threshold = max(jump_threshold, MIN_DEGREE_JUMP)

                jump_mask = abs_diff > effective_jump_threshold
                for idx in df.index[jump_mask.fillna(False)]:
                    issues.append(
                        {
                            "index": int(idx),
                            "timestamp": float(ts.loc[idx]) if not pd.isna(ts.loc[idx]) else np.nan,
                            "column": col,
                            "value": float(values.loc[idx]) if not pd.isna(values.loc[idx]) else np.nan,
                            "kind": "abrupt_jump",
                            "severity": float(abs_diff.loc[idx] / effective_jump_threshold),
                            "details": (
                                f"Sample-to-sample jump={float(abs_diff.loc[idx]):.6f} "
                                f"(threshold={float(effective_jump_threshold):.6f})."
                            ),
                        }
                    )

    return issues


def print_summary(df: pd.DataFrame, issues: pd.DataFrame) -> None:
    print("\n=== IMU Statistical Summary ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    print("\n--- Per-Column Stats ---")
    for col in NUMERIC_COLUMNS:
        stats = summarize_column(df[col])
        print(
            f"{col:>10} | count={int(stats['count'])} missing={int(stats['missing'])} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} q01={stats['q01']:.6f} median={stats['median']:.6f} "
            f"q99={stats['q99']:.6f} max={stats['max']:.6f} mad={stats['mad']:.6f}"
        )

    print("\n--- Issue Counts ---")
    if issues.empty:
        print("No outliers or unusual data detected with current thresholds.")
        return

    kind_counts = issues["kind"].value_counts()
    for kind, count in kind_counts.items():
        print(f"{kind:>22}: {int(count)}")

    print("\n--- Most Affected Channels ---")
    channel_counts = issues[issues["column"] != "timestamp"]["column"].value_counts().head(10)
    for col, count in channel_counts.items():
        print(f"{col:>10}: {int(count)}")

    print("\n--- First 20 Events By Timestamp ---")
    preview_cols = ["index", "timestamp", "column", "kind", "value", "severity", "details"]
    top = issues.sort_values(["timestamp", "index", "severity"], ascending=[True, True, False]).head(20)
    print(top[preview_cols].to_string(index=False))


def load_imu_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def run_analysis(input_csv: Path, output_csv: Path) -> None:
    df = load_imu_csv(input_csv)

    issues = detect_time_issues(df)
    issues.extend(detect_signal_issues(df))

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        issues_df = issues_df.sort_values(["timestamp", "index", "severity"], ascending=[True, True, False])

    print_summary(df, issues_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    issues_df.to_csv(output_csv, index=False)
    print(f"\nSaved issue report to: {output_csv}")


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).with_name("imu_orientation.csv")
    default_output = Path(__file__).with_name("imu_orientation_issues.csv")

    parser = argparse.ArgumentParser(
        description="Statistical analysis of IMU orientation data with outlier and unusual-data detection."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Path to imu_orientation.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to write flagged issues CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
