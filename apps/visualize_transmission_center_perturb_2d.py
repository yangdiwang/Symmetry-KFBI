#!/usr/bin/env python3
"""Plot the 2D transmission center-perturbation convergence ensemble."""

from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kfbim-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_samples(out_dir: Path) -> np.ndarray:
    path = out_dir / "samples.csv"
    if not path.is_file():
        raise RuntimeError(f"missing samples CSV: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"empty samples CSV: {path}")
    return np.atleast_1d(data)


def finite_converged(data: np.ndarray) -> np.ndarray:
    return data[(data["converged"] > 0.5) & np.isfinite(data["max_err"])]


def plot_convergence(out_dir: Path, data: np.ndarray) -> Path:
    clean = finite_converged(data)
    levels = np.array(sorted(set(clean["N"].astype(int))))
    if levels.size == 0:
        raise RuntimeError("no converged finite rows to plot")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    for sample in sorted(set(clean["sample"].astype(int))):
        rows = clean[clean["sample"].astype(int) == sample]
        rows = np.sort(rows, order="N")
        ax.loglog(
            rows["N"],
            rows["max_err"],
            color="#8d99ae",
            linewidth=0.8,
            alpha=0.28,
        )

    med = []
    q25 = []
    q75 = []
    vmin = []
    vmax = []
    for level in levels:
        vals = clean[clean["N"].astype(int) == level]["max_err"]
        med.append(np.median(vals))
        q25.append(np.quantile(vals, 0.25))
        q75.append(np.quantile(vals, 0.75))
        vmin.append(np.min(vals))
        vmax.append(np.max(vals))

    med = np.asarray(med)
    q25 = np.asarray(q25)
    q75 = np.asarray(q75)
    vmin = np.asarray(vmin)
    vmax = np.asarray(vmax)

    ax.fill_between(levels, vmin, vmax, color="#d8dee9", alpha=0.45, label="min-max")
    ax.fill_between(levels, q25, q75, color="#7aa6c2", alpha=0.35, label="25-75%")
    ax.loglog(levels, med, "o-", color="#1f4e79", linewidth=2.2, label="median")

    if levels.size >= 2:
        ref_x = np.array([levels[0], levels[-1]], dtype=float)
        ref_y2 = med[0] * (ref_x / ref_x[0]) ** -2.0
        ref_y3 = med[0] * (ref_x / ref_x[0]) ** -3.0
        ax.loglog(ref_x, ref_y2, "--", color="#4a4a4a", linewidth=1.2, label="O(h^2)")
        ax.loglog(ref_x, ref_y3, ":", color="#111111", linewidth=1.5, label="O(h^3)")

    ax.set_xlabel("N")
    ax.set_ylabel("max-norm error")
    ax.set_title("2D transmission convergence under random center perturbations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    path = out_dir / "convergence_ensemble.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_order_histogram(out_dir: Path, data: np.ndarray) -> Path:
    clean = finite_converged(data)
    levels = np.array(sorted(set(clean["N"].astype(int))))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)

    if levels.size >= 2:
        final_level = levels[-1]
        orders = clean[
            (clean["N"].astype(int) == final_level) & np.isfinite(clean["order"])
        ]["order"]
        ax.hist(orders, bins=min(16, max(6, len(orders) // 3)), color="#2f6f8f", alpha=0.82)
        ax.axvline(np.median(orders), color="#111111", linewidth=1.6, label="median")
        ax.set_title(f"Observed order into N={final_level}")
        ax.set_xlabel("log2(error_previous / error_current)")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "need at least two levels", ha="center", va="center")
        ax.set_axis_off()

    path = out_dir / "order_histogram.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_center_offsets(out_dir: Path, data: np.ndarray) -> Path:
    clean = finite_converged(data)
    levels = np.array(sorted(set(clean["N"].astype(int))))
    final_level = levels[-1]
    final_rows = clean[clean["N"].astype(int) == final_level]

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    sc = ax.scatter(
        final_rows["cx"],
        final_rows["cy"],
        c=final_rows["max_err"],
        s=34,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
    )
    ax.scatter([0.07], [-0.04], marker="+", s=140, color="#d62828", linewidths=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("center x")
    ax.set_ylabel("center y")
    ax.set_title(f"Random centers colored by N={final_level} error")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="max-norm error")

    path = out_dir / "center_offsets.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: visualize_transmission_center_perturb_2d.py OUTPUT_DIR", file=sys.stderr)
        return 2

    out_dir = Path(argv[1])
    if not out_dir.is_dir():
        print(f"output directory not found: {out_dir}", file=sys.stderr)
        return 2

    data = load_samples(out_dir)
    paths = [
        plot_convergence(out_dir, data),
        plot_order_histogram(out_dir, data),
        plot_center_offsets(out_dir, data),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
