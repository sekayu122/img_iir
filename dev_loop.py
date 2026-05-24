#!/usr/bin/env python3
"""Run one development evaluation and validation gate.

暗部ノイズ低減などのアルゴリズム開発時に、run_experiment.py を実行し、
train評価とvalidation評価を履歴として保存します。best更新は、train scoreが改善し、
かつconfig.yamlに定義したvalidationがすべて閾値以上の場合だけ行います。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from iir_filters import available_filter_names


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
HISTORY_DIR = ROOT / "output" / "dev_history"
BEST_ROOT = ROOT / "output" / "best_ai_filter"
TMP_CONFIG_DIR = ROOT / "output" / "dev_tmp_configs"
REQUIRED_VALIDATIONS = ("validation_synthetic", "validation_skimage")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train and validation experiments using config.yaml as the base settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filter_name",
        choices=available_filter_names(),
        help="Filter class name to test.",
    )
    parser.add_argument(
        "experiment_name",
        help="Name used to separate run_dir, validation run_dir, history, and best outputs.",
    )
    return parser


def slugify(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    slug = slug.strip("._-")
    if not slug:
        raise RuntimeError("experiment name must contain at least one alphanumeric character")
    return slug


def prepare_config(base_config: dict[str, Any], filter_name: str, experiment_name: str) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["filter"] = {"name": filter_name}

    experiment_id = slugify(experiment_name)
    experiment_root = Path("output") / "experiments" / experiment_id
    config["experiment_name"] = experiment_id
    config["run_dir"] = str(experiment_root / "train")

    validations = config.get("validations", []) or []
    if not isinstance(validations, list):
        raise RuntimeError("config value 'validations' must be a list")
    for validation in validations:
        if not isinstance(validation, dict):
            raise RuntimeError("validation entries must be mappings")
        name = str(validation.get("name", "validation"))
        validation["run_dir"] = str(experiment_root / name)
    return config


def run_command(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed.stdout


def load_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(raw, dict):
        raise RuntimeError(f"config must be a mapping: {CONFIG_PATH}")
    return raw


def path_from_value(value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"config value '{key}' must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics not found: {metrics_path}")
    return json.loads(metrics_path.read_text())


def read_total_score(run_dir: Path) -> float:
    return float(read_metrics(run_dir)["total_score"])


def best_dir_for_run(run_dir: Path) -> Path:
    if run_dir.name == "train" and run_dir.parent.name:
        return BEST_ROOT / run_dir.parent.name
    return BEST_ROOT / run_dir.name


def read_best_score(best_dir: Path) -> float:
    path = best_dir / "best_score.txt"
    if not path.exists():
        return float("-inf")
    return float(path.read_text().strip())


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_run_artifacts(run_dir: Path, dst_dir: Path) -> None:
    for name in (
        "metrics.json",
        "summary.txt",
        "config.json",
        "evaluate_stdout.json",
        "apply_stdout.txt",
        "apply_stderr.txt",
        "evaluate_stderr.txt",
    ):
        copy_if_exists(run_dir / name, dst_dir / name)


def write_temp_config(name: str, config: dict[str, Any]) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def run_experiment_with_config(config_path: Path, filter_name: str, experiment_name: str) -> str:
    return run_command(
        [sys.executable, "run_experiment.py", filter_name, experiment_name],
        {"RUN_EXPERIMENT_CONFIG": str(config_path)},
    )


def validation_config(base_config: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_config)
    for key in ("input", "run_dir", "gt", "eval_before", "overwrite"):
        if key in validation:
            merged[key] = validation[key]
    if "filter" in validation:
        merged["filter"] = validation["filter"]
    if "evaluation" in validation:
        inherited = copy.deepcopy(base_config.get("evaluation", {}))
        override = validation["evaluation"] or {}
        if not isinstance(override, dict):
            raise RuntimeError("validation.evaluation must be a mapping")
        inherited.update(override)
        merged["evaluation"] = inherited
    merged.pop("validations", None)
    return merged


def run_validations(base_config: dict[str, Any], filter_name: str, experiment_name: str) -> list[dict[str, Any]]:
    validations = base_config.get("validations", []) or []
    if not isinstance(validations, list):
        raise RuntimeError("config value 'validations' must be a list")

    results: list[dict[str, Any]] = []
    for index, validation in enumerate(validations):
        if not isinstance(validation, dict):
            raise RuntimeError("validation entries must be mappings")
        name = str(validation.get("name", f"validation_{index}"))
        config = validation_config(base_config, validation)
        config_path = write_temp_config(name, config)
        print(f"running validation: {name}")
        print(run_experiment_with_config(config_path, filter_name, experiment_name))

        run_dir = path_from_value(config.get("run_dir"), f"validations[{index}].run_dir")
        score = read_total_score(run_dir)
        min_score = float(validation.get("min_total_score", 0.0))
        results.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "score": score,
                "min_total_score": min_score,
                "passed": score >= min_score,
            }
        )
    return results


def archive_trial(run_dir: Path, score: float, validation_results: list[dict[str, Any]]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = HISTORY_DIR / f"{stamp}_score_{score:.2f}"
    trial_dir.mkdir(parents=True, exist_ok=False)

    copy_run_artifacts(run_dir, trial_dir)
    (trial_dir / "validation_results.json").write_text(
        json.dumps(validation_results, indent=2, ensure_ascii=False) + "\n"
    )
    for result in validation_results:
        copy_run_artifacts(Path(result["run_dir"]), trial_dir / "validations" / result["name"])

    diff = run_command(["git", "diff", "--", "iir_filters.py"])
    (trial_dir / "iir_filters.diff").write_text(diff)
    shutil.copy2(ROOT / "iir_filters.py", trial_dir / "iir_filters.py")
    return trial_dir


def validation_gate_passed(validation_results: list[dict[str, Any]]) -> bool:
    by_name = {str(result["name"]): result for result in validation_results}
    return all(
        name in by_name and bool(by_name[name]["passed"]) for name in REQUIRED_VALIDATIONS
    )


def print_validation_results(validation_results: list[dict[str, Any]]) -> None:
    by_name = {str(result["name"]): result for result in validation_results}
    for name in REQUIRED_VALIDATIONS:
        result = by_name.get(name)
        if result is None:
            print(f"  FAIL {name}: validation is not configured")
            continue
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {status} {result['name']}: "
            f"score={result['score']:.2f}, min={result['min_total_score']:.2f}"
        )


def update_best_if_needed(run_dir: Path, score: float, validation_results: list[dict[str, Any]]) -> bool:
    best_dir = best_dir_for_run(run_dir)
    best_score = read_best_score(best_dir)
    if score <= best_score:
        print(f"score: {score:.2f}, best: {best_score:.2f}")
        return False
    if not validation_gate_passed(validation_results):
        print("validation gate failed; best was not updated")
        print_validation_results(validation_results)
        return False

    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "iir_filters.py", best_dir / "iir_filters.py")
    copy_run_artifacts(run_dir, best_dir)
    (best_dir / "validation_results.json").write_text(
        json.dumps(validation_results, indent=2, ensure_ascii=False) + "\n"
    )
    for result in validation_results:
        copy_run_artifacts(Path(result["run_dir"]), best_dir / "validations" / result["name"])
    (best_dir / "best_score.txt").write_text(f"{score:.6f}\n")
    print(f"NEW BEST: {score:.2f} (previous: {best_score:.2f})")
    return True


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = prepare_config(load_config(), args.filter_name, args.experiment_name)
    except RuntimeError as exc:
        parser.error(str(exc))

    train_config_path = write_temp_config("train", config)
    print(run_experiment_with_config(train_config_path, args.filter_name, args.experiment_name))

    run_dir = path_from_value(config.get("run_dir"), "run_dir")
    score = read_total_score(run_dir)
    validation_results = run_validations(config, args.filter_name, args.experiment_name)
    trial_dir = archive_trial(run_dir, score, validation_results)
    print(f"archived: {trial_dir}")
    update_best_if_needed(run_dir, score, validation_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
