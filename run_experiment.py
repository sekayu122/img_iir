#!/usr/bin/env python3
"""Run one filter experiment and save apply/evaluate logs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from iir_filters import available_filter_names


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """コマンドを実行し、失敗時は標準出力/標準エラーを含めて例外にする。"""
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a filter, evaluate it against GT, and save experiment artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input TIFF directory passed to apply_filter.py")
    parser.add_argument("run_dir", type=Path, help="new experiment output directory")
    parser.add_argument("gt", type=Path, help="ground-truth TIFF directory passed to evaluate_img_quarity.py")
    parser.add_argument(
        "--eval-before",
        type=Path,
        default=None,
        help="before directory for evaluation; defaults to input",
    )
    parser.add_argument(
        "--filter",
        choices=available_filter_names(),
        default="AIExpFilter",
        help="filter algorithm passed to apply_filter.py",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove run_dir before running if it already exists",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    repo_dir = Path(__file__).resolve().parent
    eval_before = args.eval_before or args.input

    for label, path in (("input", args.input), ("eval-before", eval_before), ("gt", args.gt)):
        if not path.is_dir():
            parser.error(f"{label} must be a directory: {path}")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be 0.0..1.0")

    if args.run_dir.exists():
        if not args.overwrite:
            parser.error(f"run_dir already exists: {args.run_dir}")
        shutil.rmtree(args.run_dir)

    filtered_dir = args.run_dir / "filtered"
    args.run_dir.mkdir(parents=True)

    config = {
        "input": str(args.input),
        "eval_before": str(eval_before),
        "filtered": str(filtered_dir),
        "gt": str(args.gt),
        "filter": args.filter,
        "alpha": args.alpha,
    }
    write_json(args.run_dir / "config.json", config)

    apply_command = [
        sys.executable,
        str(repo_dir / "apply_filter.py"),
        str(args.input),
        str(filtered_dir),
        "--filter",
        args.filter,
        "--alpha",
        str(args.alpha),
    ]
    evaluate_command = [
        sys.executable,
        str(repo_dir / "evaluate_img_quarity.py"),
        str(eval_before),
        str(filtered_dir),
        str(args.gt),
        "--json",
    ]

    try:
        apply_result = run_command(apply_command, repo_dir)
        evaluate_result = run_command(evaluate_command, repo_dir)
        metrics = json.loads(evaluate_result.stdout)
    except Exception:
        (args.run_dir / "failed.txt").write_text("experiment failed\n")
        raise

    (args.run_dir / "apply_stdout.txt").write_text(apply_result.stdout)
    (args.run_dir / "apply_stderr.txt").write_text(apply_result.stderr)
    (args.run_dir / "evaluate_stdout.json").write_text(evaluate_result.stdout)
    (args.run_dir / "evaluate_stderr.txt").write_text(evaluate_result.stderr)
    write_json(args.run_dir / "metrics.json", metrics)

    image_quality = metrics.get("image_quality", {})
    summary = [
        f"total_score: {metrics['total_score']:.2f}",
        f"overall_psnr_db: {image_quality.get('psnr_db', float('nan')):.3f}",
        f"overall_ssim: {image_quality.get('ssim', float('nan')):.6f}",
        f"filter: {args.filter}",
        f"filtered: {filtered_dir}",
        "",
        "band,noise,blur,motion,psnr_score,psnr_db,ssim_score,ssim",
    ]
    for band_name, band in metrics["bands"].items():
        summary.append(
            f"{band_name},{band['noise_score']:.2f},{band['blur_score']:.2f},"
            f"{band['motion_score']:.2f},{band['psnr_score']:.2f},"
            f"{band['psnr_db']:.3f},{band['ssim_score']:.2f},{band['ssim']:.6f}"
        )
    (args.run_dir / "summary.txt").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
