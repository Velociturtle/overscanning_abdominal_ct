# BLOCK 6 – Figure generation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .config import NIFTI_DIR, CSV_PATH


def save_fig(fig, name: str, out_dir: Path = NIFTI_DIR, exts=("png",), dpi=300, close=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def safe_ylim(ax, upper):
    if np.isfinite(upper) and upper > 0:
        ax.set_ylim(-upper * 1.05, upper * 1.05)


def generate_all() -> None:
    """Generate scatter, box, and bar plots from the results CSV."""
    df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    cranial = pd.to_numeric(
        df.get("calc_cranial_overscan_mm", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).values
    caudal = pd.to_numeric(
        df.get("calc_caudal_overscan_mm", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).values
    x_idx = np.arange(len(cranial))

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x_idx, cranial, s=10, label="Cranial")
    ax.scatter(x_idx, -caudal, s=10, label="Caudal")
    max_abs = np.nanmax(
        [np.abs(cranial).max() if cranial.size else 0, np.abs(caudal).max() if caudal.size else 0]
    )
    safe_ylim(ax, max_abs)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Case index")
    ax.set_ylabel("Overscan excess (mm)\n(+ cranial / − caudal)")
    ax.set_title("Cranial vs Caudal Overscan Excess")
    ax.legend(frameon=False)
    plt.tight_layout()
    save_fig(fig, "scatter_cranial_caudal")
    print(f"Scatterplot saved to {NIFTI_DIR / 'scatter_cranial_caudal.png'}")

    # Boxplot
    box_cols = [
        "calc_cranial_overscan_mm",
        "calc_caudal_overscan_mm",
        "calc_total_overscan_mm",
    ]
    box_labels = ["Cranial", "Caudal", "Total"]
    data = [
        pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce").dropna().values
        for c in box_cols
    ]

    fig, ax = plt.subplots(figsize=(5, 7))
    if any(len(d) for d in data):
        bp = ax.boxplot(
            data,
            vert=True,
            whis=1.5,
            showmeans=True,
            meanline=True,
            showcaps=True,
            showfliers=True,
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            meanprops=dict(color="black", linestyle="--", linewidth=1),
            whiskerprops=dict(color="black", linestyle="--", linewidth=1),
            capprops=dict(color="black", linewidth=1),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor="none",
                markeredgecolor="black",
                alpha=0.8,
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_edgecolor("black")

    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels)
    ax.set_ylabel("Overscan excess (mm)")
    ax.set_title("Overscan Excess – Box & Whisker")
    plt.tight_layout()
    save_fig(fig, "box_cranial_caudal_total")
    print(f"Box-and-whisker plot saved to {NIFTI_DIR / 'box_cranial_caudal_total.png'}")

    # Bar
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.bar(x_idx, cranial, width=0.8, color=colors[0], label="Cranial")
    ax.bar(x_idx, -caudal, width=0.8, color=colors[1], label="Caudal")
    max_abs = np.nanmax(
        [np.abs(cranial).max() if cranial.size else 0, np.abs(caudal).max() if caudal.size else 0]
    )
    safe_ylim(ax, max_abs)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Case index")
    ax.set_ylabel("Overscan excess (mm)\n(+ cranial / − caudal)")
    ax.set_title("Cranial vs Caudal Overscan Excess – Bar Plot")
    ax.legend(frameon=False)
    plt.tight_layout()
    save_fig(fig, "bar_cranial_caudal")
    print(f"Bar plot saved to {NIFTI_DIR / 'bar_cranial_caudal.png'}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    generate_all()
