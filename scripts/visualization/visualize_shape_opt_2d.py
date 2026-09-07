#!/usr/bin/env python3
"""Render frames and summary plots for the 2D transmission shape optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"empty CSV: {path}")
    return np.atleast_1d(data)


def load_boundary(path: Path):
    data = load_csv(path)
    return np.column_stack((data["x"], data["y"]))


def boundary_path(out_dir: Path, iteration: int) -> Path:
    return out_dir / f"boundary_iter_{iteration:04d}.csv"


def available_boundaries(out_dir: Path, summary):
    curves = []
    for row in summary:
        iteration = int(row["iter"])
        path = boundary_path(out_dir, iteration)
        if path.is_file():
            curves.append((iteration, load_boundary(path), row))
    if not curves:
        raise RuntimeError("no boundary_iter_XXXX.csv files found")
    return curves


def axis_limits(target, curves):
    xs = [target[:, 0]]
    ys = [target[:, 1]]
    for _, curve, _ in curves:
        xs.append(curve[:, 0])
        ys.append(curve[:, 1])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    cx = 0.5 * (float(x.min()) + float(x.max()))
    cy = 0.5 * (float(y.min()) + float(y.max()))
    radius = 0.58 * max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    return (cx - radius, cx + radius), (cy - radius, cy + radius)


def plot_objective(out_dir: Path, summary) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.semilogy(summary["iter"], summary["objective"], marker="o", label="objective")
    ax.semilogy(summary["iter"], summary["data_misfit"], marker="s", label="data")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title("Shape optimization objective")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "objective_history.png", dpi=180)
    plt.close(fig)


def plot_shape_evolution(out_dir: Path, target, curves) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(curves) - 1)
    ax.plot(target[:, 0], target[:, 1], "k--", linewidth=2.0, label="target")
    for idx, (iteration, curve, _) in enumerate(curves):
        color = cmap(idx / denom)
        linewidth = 1.2 if idx not in (0, len(curves) - 1) else 2.0
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=linewidth,
                label=f"iter {iteration}" if idx in (0, len(curves) - 1) else None)
    xlim, ylim = axis_limits(target, curves)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary evolution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "shape_evolution.png", dpi=180)
    plt.close(fig)


def plot_frame(path: Path, target, curve, row, xlim, ylim) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    ax.plot(target[:, 0], target[:, 1], "k--", linewidth=2.0, label="target")
    ax.plot(curve[:, 0], curve[:, 1], color="#2271b2", linewidth=2.3, label="current")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"iter {int(row['iter'])}   J={float(row['objective']):.3e}   "
        f"data={float(row['data_misfit']):.3e}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_frames_and_gif(out_dir: Path, target, curves) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("shape_iter_*.png"):
        old.unlink()

    xlim, ylim = axis_limits(target, curves)
    frame_paths = []
    for iteration, curve, row in curves:
        frame_path = frames_dir / f"shape_iter_{iteration:04d}.png"
        plot_frame(frame_path, target, curve, row, xlim, ylim)
        frame_paths.append(frame_path)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write shape_evolution.gif") from exc

    images = [Image.open(path) for path in frame_paths]
    try:
        images[0].save(
            out_dir / "shape_evolution.gif",
            save_all=True,
            append_images=images[1:],
            duration=550,
            loop=0,
        )
    finally:
        for image in images:
            image.close()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: visualize_shape_opt_2d.py OUTPUT_DIR", file=sys.stderr)
        return 2

    out_dir = Path(argv[1])
    if not out_dir.is_dir():
        print(f"output directory not found: {out_dir}", file=sys.stderr)
        return 2

    summary_path = out_dir / "summary.csv"
    target_path = out_dir / "target_boundary.csv"
    if not summary_path.is_file():
        print(f"summary not found: {summary_path}", file=sys.stderr)
        return 2
    if not target_path.is_file():
        print(f"target boundary not found: {target_path}", file=sys.stderr)
        return 2

    summary = load_csv(summary_path)
    target = load_boundary(target_path)
    curves = available_boundaries(out_dir, summary)

    plot_objective(out_dir, summary)
    plot_shape_evolution(out_dir, target, curves)
    write_frames_and_gif(out_dir, target, curves)
    print(f"wrote visualizations to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
