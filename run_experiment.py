#!/usr/bin/env python3
"""Run one filter experiment from config.yaml and save apply/evaluate logs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from iir_filters import available_filter_names


CONFIG_PATH = Path(os.environ.get("RUN_EXPERIMENT_CONFIG", "config.yaml"))


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


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise RuntimeError(f"could not parse YAML config: {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"config must be a mapping: {path}")
    return raw


def path_from_config(value: Any, repo_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"config value '{key}' must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else repo_dir / path


def optional_path_from_config(value: Any, repo_dir: Path, key: str) -> Path | None:
    if value is None or value == "":
        return None
    return path_from_config(value, repo_dir, key)


def filter_name_from_mapping(config: dict[str, Any]) -> str:
    filter_config = config.get("filter", {})
    if isinstance(filter_config, str):
        filter_name = filter_config
    elif isinstance(filter_config, dict):
        filter_name = str(filter_config.get("name", "AIExpFilter"))
    else:
        raise RuntimeError("config value 'filter' must be a string or mapping")
    if filter_name not in available_filter_names():
        choices = ", ".join(available_filter_names())
        raise RuntimeError(f"unknown filter: {filter_name}; choices: {choices}")
    return filter_name


def evaluation_options_from_config(config: dict[str, Any]) -> list[str]:
    evaluation = config.get("evaluation", {})
    if evaluation is None:
        return []
    if not isinstance(evaluation, dict):
        raise RuntimeError("config value 'evaluation' must be a mapping")

    option_names = {
        "target_noise_db": "--target-noise-db",
        "blur_min_ratio": "--blur-min-ratio",
        "motion_error_ref": "--motion-error-ref",
        "motion_threshold": "--motion-threshold",
        "noise_floor": "--noise-floor",
        "psnr_min_db": "--psnr-min-db",
        "psnr_target_db": "--psnr-target-db",
        "ssim_min": "--ssim-min",
        "ssim_target": "--ssim-target",
        "min_pixels": "--min-pixels",
    }
    args: list[str] = []
    for key, option in option_names.items():
        if key in evaluation and evaluation[key] is not None:
            args.extend([option, str(evaluation[key])])

    for roi in evaluation.get("dark_rois", []) or []:
        if isinstance(roi, str):
            roi_text = roi
        elif isinstance(roi, dict):
            roi_text = f"{roi['x']},{roi['y']},{roi['width']},{roi['height']}"
        else:
            raise RuntimeError("dark_rois entries must be strings or mappings")
        args.extend(["--dark-roi", roi_text])

    weights = evaluation.get("weights", {}) or {}
    if isinstance(weights, dict):
        for name, weight in weights.items():
            args.extend(["--weight", f"{name}={weight}"])
    elif isinstance(weights, list):
        for item in weights:
            args.extend(["--weight", str(item)])
    else:
        raise RuntimeError("evaluation.weights must be a mapping or list")
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a filter and evaluate it using config.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    parser.parse_args()
    repo_dir = Path(__file__).resolve().parent

    try:
        config_path = CONFIG_PATH if CONFIG_PATH.is_absolute() else repo_dir / CONFIG_PATH
        config = load_yaml_config(config_path)
        input_dir = path_from_config(config.get("input"), repo_dir, "input")
        run_dir = path_from_config(config.get("run_dir"), repo_dir, "run_dir")
        gt_dir = path_from_config(config.get("gt"), repo_dir, "gt")
        eval_before = optional_path_from_config(config.get("eval_before"), repo_dir, "eval_before") or input_dir
        filter_name = filter_name_from_mapping(config)
        overwrite = bool(config.get("overwrite", False))
        evaluate_args = evaluation_options_from_config(config)
    except RuntimeError as exc:
        parser.error(str(exc))

    for label, path in (("input", input_dir), ("eval_before", eval_before), ("gt", gt_dir)):
        if not path.is_dir():
            parser.error(f"{label} must be a directory: {path}")

    if run_dir.exists():
        if not overwrite:
            parser.error(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)

    filtered_dir = run_dir / "filtered"
    run_dir.mkdir(parents=True)

    resolved_config = dict(config)
    resolved_config.update(
        {
            "input": str(input_dir),
            "eval_before": str(eval_before),
            "filtered": str(filtered_dir),
            "gt": str(gt_dir),
            "filter": {"name": filter_name},
            "overwrite": overwrite,
        }
    )
    write_json(run_dir / "config.json", resolved_config)

    apply_command = [
        sys.executable,
        str(repo_dir / "apply_filter.py"),
        str(input_dir),
        str(filtered_dir),
        "--filter",
        filter_name,
    ]
    evaluate_command = [
        sys.executable,
        str(repo_dir / "evaluate_img_quarity.py"),
        str(eval_before),
        str(filtered_dir),
        str(gt_dir),
        "--json",
        *evaluate_args,
    ]

    try:
        apply_result = run_command(apply_command, repo_dir)
        evaluate_result = run_command(evaluate_command, repo_dir)
        metrics = json.loads(evaluate_result.stdout)
    except Exception:
        (run_dir / "failed.txt").write_text("experiment failed\n")
        raise

    (run_dir / "apply_stdout.txt").write_text(apply_result.stdout)
    (run_dir / "apply_stderr.txt").write_text(apply_result.stderr)
    (run_dir / "evaluate_stdout.json").write_text(evaluate_result.stdout)
    (run_dir / "evaluate_stderr.txt").write_text(evaluate_result.stderr)
    write_json(run_dir / "metrics.json", metrics)

    image_quality = metrics.get("image_quality", {})
    summary = [
        f"total_score: {metrics['total_score']:.2f}",
        f"overall_psnr_db: {image_quality.get('psnr_db', float('nan')):.3f}",
        f"overall_ssim: {image_quality.get('ssim', float('nan')):.6f}",
        f"filter: {filter_name}",
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
    (run_dir / "summary.txt").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
