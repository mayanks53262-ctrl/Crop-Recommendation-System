from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
REPORT_PATH = ROOT / "reports" / "dataset_audit.txt"

EXPECTED_FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]
TARGET = "label"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Keep the filename "
            "'Crop_recommendation.csv' inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    lines = []
    lines.append("CROP RECOMMENDATION DATASET AUDIT")
    lines.append("=" * 38)
    lines.append(f"File: {DATA_PATH.relative_to(ROOT)}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append(f"Column names: {list(df.columns)}")
    lines.append("")

    missing_columns = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
    extra_columns = [c for c in df.columns if c not in EXPECTED_FEATURES + [TARGET]]
    lines.append(f"Missing expected columns: {missing_columns}")
    lines.append(f"Unexpected columns: {extra_columns}")

    lines.append("")
    lines.append("Missing values by column:")
    lines.append(df.isna().sum().to_string())

    lines.append("")
    lines.append(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    numeric_cols = [c for c in EXPECTED_FEATURES if c in df.columns]
    lines.append("")
    lines.append("Numeric summary:")
    lines.append(df[numeric_cols].describe().round(3).to_string())

    if TARGET in df.columns:
        counts = df[TARGET].value_counts(dropna=False).sort_index()
        lines.append("")
        lines.append(f"Unique target classes: {df[TARGET].nunique(dropna=True)}")
        lines.append("Class distribution:")
        lines.append(counts.to_string())

    # Range checks are warnings for review, not automatic data deletion.
    bounds = {
        "N": (0, 300), "P": (0, 300), "K": (0, 500),
        "temperature": (-10, 60), "humidity": (0, 100),
        "ph": (0, 14), "rainfall": (0, 5000)
    }
    lines.append("")
    lines.append("Range review (broad sanity bounds; review flags, do not auto-delete):")
    for col, (low, high) in bounds.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad = ((numeric < low) | (numeric > high) | numeric.isna())
            lines.append(f"{col}: flagged {int(bad.sum())} row(s), expected broad range [{low}, {high}]")

    lines.append("")
    lines.append("Interpretation note:")
    lines.append(
        "This report audits structure and basic quality only. It does not establish "
        "real-world agricultural validity or geographic generalization."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
