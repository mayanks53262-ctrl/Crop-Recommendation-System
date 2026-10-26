from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.txt"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def save_plot(fig, filename: str) -> None:
    """Save a matplotlib figure into the reports/figures folder."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = sns.color_palette("viridis", len(counts))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Crop Class Distribution")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90)
    save_plot(fig, "class_distribution.png")


def plot_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, FEATURES):
        sns.histplot(df[feature], bins=30, kde=True, color="steelblue", ax=ax)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

    for ax in axes[len(FEATURES):]:
        ax.set_visible(False)

    save_plot(fig, "feature_histograms.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, FEATURES):
        sns.boxplot(y=df[feature], color="lightcoral", ax=ax)
        ax.set_title(f"{feature} Boxplot")
        ax.set_ylabel(feature)

    for ax in axes[len(FEATURES):]:
        ax.set_visible(False)

    save_plot(fig, "feature_boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")


def plot_feature_distributions_by_crop(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()
    palette = sns.color_palette("viridis", df[TARGET].nunique())

    for ax, feature in zip(axes, FEATURES):
        sns.boxplot(
            data=df,
            x=TARGET,
            y=feature,
            hue=TARGET,
            palette=palette,
            dodge=False,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{feature} by Crop")
        ax.set_xlabel("Crop")
        ax.set_ylabel(feature)
        ax.tick_params(axis="x", rotation=90)

    for ax in axes[len(FEATURES):]:
        ax.set_visible(False)

    save_plot(fig, "feature_distributions_by_crop.png")


def write_summary(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts()
    numeric_summary = df[FEATURES].describe().round(2)
    correlations = df[FEATURES].corr().abs().where(
        ~pd.DataFrame(
            [[left == right for right in FEATURES] for left in FEATURES],
            index=FEATURES,
            columns=FEATURES,
        )
    )
    strongest_pair = correlations.stack().idxmax()
    strongest_correlation = correlations.stack().max()
    iqr_flags = {}
    for feature in FEATURES:
        lower_quartile = df[feature].quantile(0.25)
        upper_quartile = df[feature].quantile(0.75)
        iqr = upper_quartile - lower_quartile
        iqr_flags[feature] = int(
            ((df[feature] < lower_quartile - 1.5 * iqr)
             | (df[feature] > upper_quartile + 1.5 * iqr)).sum()
        )
    highest_iqr_feature = max(iqr_flags, key=iqr_flags.get)

    lines = []
    lines.append("Crop Recommendation System - Exploratory Data Analysis Summary")
    lines.append("=" * 68)
    lines.append(f"Dataset shape: {df.shape}")
    lines.append(f"Crop classes: {df[TARGET].nunique()}")
    lines.append(f"Balanced sample pattern: {counts.min()} to {counts.max()} rows per crop")
    lines.append("")
    lines.append("Key observations:")
    lines.append(
        f"- The target is balanced: each of the {df[TARGET].nunique()} crops has "
        f"{counts.min()} to {counts.max()} observations."
    )
    lines.append(
        f"- The strongest absolute pairwise feature correlation is between "
        f"{strongest_pair[0]} and {strongest_pair[1]} "
        f"({strongest_correlation:.2f}); correlation does not imply causation."
    )
    lines.append(
        f"- The 1.5 x IQR review rule flags the most values for {highest_iqr_feature} "
        f"({iqr_flags[highest_iqr_feature]} rows). These are candidates for review, "
        "not automatically invalid observations."
    )
    lines.append(
        "- The crop-wise boxplots compare the spread of each numeric feature across "
        "crop labels; visible differences may help distinguish classes."
    )
    lines.append(
        "- These plots describe this dataset only and do not establish that its "
        "recommendations generalize to real farms or other regions."
    )
    lines.append("")
    lines.append("Numeric feature summary:")
    lines.append(numeric_summary.to_string())

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions_by_crop(df)
    write_summary(df)

    print(f"EDA complete. Figures saved in: {FIGURES_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
