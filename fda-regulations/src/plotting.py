from pathlib import Path
import logging
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def has_required_columns(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Return True if the DataFrame contains all required columns."""
    return all(column in df.columns for column in required_columns)


def format_axis_labels(
    labels: pd.Series,
    max_chars: int = 28,
    wrap_width: int = 16,
) -> list[str]:
    """Shorten + wrap long labels for readability."""
    formatted: list[str] = []

    for raw_label in labels.astype(str):
        text = raw_label.strip()

        if len(text) > max_chars:
            text = f"{text[:max_chars - 3]}..."

        wrapped = "\n".join(
            textwrap.wrap(text, width=wrap_width, break_long_words=False)
        )

        formatted.append(wrapped if wrapped else text)

    return formatted


def plot_line(
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Write a line chart if required columns are present."""
    if df.empty:
        return

    if not has_required_columns(df, [x_key, y_key]):
        logger.warning(
            "Skipping line chart %s because required columns are missing: %s, %s",
            output_path,
            x_key,
            y_key,
        )
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df[x_key].astype(str), df[y_key].astype(float), marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved line chart to %s", output_path)


def plot_bar(
    df: pd.DataFrame,
    label_key: str,
    value_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    top_n: int | None = None,
    max_label_chars: int = 28,
    wrap_width: int = 16,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Write a bar chart if required columns are present."""
    if df.empty:
        return

    if not has_required_columns(df, [label_key, value_key]):
        logger.warning(
            "Skipping bar chart %s because required columns are missing: %s, %s",
            output_path,
            label_key,
            value_key,
        )
        return

    plot_df = df.head(top_n) if top_n is not None else df

    labels = format_axis_labels(
        plot_df[label_key],
        max_chars=max_label_chars,
        wrap_width=wrap_width,
    )

    plt.figure(figsize=figsize)
    plt.bar(labels, plot_df[value_key].astype(float))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved bar chart to %s", output_path)


def plot_barh(
    df: pd.DataFrame,
    label_key: str,
    value_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    top_n: int | None = None,
    max_label_chars: int = 40,
    wrap_width: int = 28,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    if df.empty:
        return

    if not has_required_columns(df, [label_key, value_key]):
        logger.warning(
            "Skipping horizontal bar chart %s because required columns are missing: %s, %s",
            output_path,
            label_key,
            value_key,
        )
        return

    plot_df = df.head(top_n) if top_n is not None else df

    labels = format_axis_labels(
        plot_df[label_key],
        max_chars=max_label_chars,
        wrap_width=wrap_width,
    )

    plt.figure(figsize=figsize)
    plt.barh(labels, plot_df[value_key].astype(float))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()