#!/usr/bin/env python3
"""Compare before/after/GT TIFF image sequences for IIR filter quality.

フィルタ適用前、適用後、GTの3つのTIFF連番フォルダを比較し、
dark/normal/highごとにノイズ残差低減、GT基準のエッジ保持、
GT基準のmotion errorを評価して合計点を出力します。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class BandDefinition:
    """輝度レンジの定義。"""

    name: str
    low: float
    high: float


@dataclass(frozen=True)
class Roi:
    """矩形ROI。"""

    x: int
    y: int
    width: int
    height: int


BANDS = (
    BandDefinition("dark", 0.00, 0.25),
    BandDefinition("normal", 0.25, 0.75),
    BandDefinition("high", 0.75, 1.01),
)


DEFAULT_SCORE_WEIGHTS = {
    "dark_noise": 1.0,
    "dark_blur": 1.0,
    "dark_motion": 1.0,
    "normal_noise": 1.0,
    "normal_blur": 1.0,
    "normal_motion": 1.0,
    "high_noise": 1.0,
    "high_blur": 1.0,
    "high_motion": 1.0,
}

DEFAULT_NOISE_FLOOR = 0.002


def list_image_files(input_dir: Path) -> list[Path]:
    """フォルダ内のTIFF画像を名前順に返す。"""
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise RuntimeError(f"no TIFF images found: {input_dir}")
    return files


def read_image(input_path: Path) -> np.ndarray:
    """画像を元のbit深度のまま読む。"""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read image: {input_path}")
    return image


def dtype_max_value(image: np.ndarray) -> float:
    """dtypeから正規化用の最大値を返す。"""
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    max_value = float(np.max(image))
    return max_value if max_value > 1.0 else 1.0


def to_luminance01(image: np.ndarray) -> np.ndarray:
    """BGR画像を0..1の輝度画像へ変換する。"""
    if image.ndim == 2:
        luma = image.astype(np.float32)
    elif image.shape[2] == 1:
        luma = image[:, :, 0].astype(np.float32)
    else:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
    return np.clip(luma / dtype_max_value(image), 0.0, 1.0).astype(np.float32)


def make_band_mask(luma01: np.ndarray, band: BandDefinition) -> np.ndarray:
    """指定輝度レンジのマスクを作る。"""
    return (luma01 >= band.low) & (luma01 < band.high)


def make_roi_mask(shape: tuple[int, int], rois: list[Roi]) -> np.ndarray:
    """ROIリストからboolマスクを作る。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for roi in rois:
        x2 = min(width, roi.x + roi.width)
        y2 = min(height, roi.y + roi.height)
        if roi.x >= width or roi.y >= height:
            continue
        mask[roi.y:y2, roi.x:x2] = True
    return mask


def validate_rois(rois: list[Roi], shape: list[int]) -> None:
    """ROIが画像範囲内にあることを確認する。"""
    height, width = shape[:2]
    for roi in rois:
        if roi.x + roi.width > width or roi.y + roi.height > height:
            raise RuntimeError(
                f"dark ROI exceeds image size: "
                f"{roi.x},{roi.y},{roi.width},{roi.height}, image={width}x{height}"
            )


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """境界の混入を減らすためマスクを縮める。"""
    if pixels <= 0:
        return mask
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """対象レンジの境界付近マスクを作る。"""
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    return dilated & ~eroded


def edge_strength(luma01: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[float, int]:
    """境界付近のSobel勾配95パーセンタイルを返す。"""
    edge_mask = boundary_mask(mask, 2)
    if int(np.count_nonzero(edge_mask)) < min_pixels:
        edge_mask = mask
    if not np.any(edge_mask):
        return float("nan"), 0

    sobel_x = cv2.Sobel(luma01, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(luma01, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    return float(np.percentile(gradient[edge_mask], 95)), int(np.count_nonzero(edge_mask))


def safe_mean(values: list[float]) -> float:
    """有限値だけの平均を返す。"""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def score_noise_reduction(
    before_noise: float,
    after_noise: float,
    target_db: float,
    noise_floor: float,
) -> tuple[float, float]:
    """ノイズ低減量[dB]とスコアを返す。"""
    if not np.isfinite(before_noise + after_noise):
        return float("nan"), float("nan")
    if before_noise <= noise_floor:
        if after_noise <= noise_floor:
            return 0.0, 100.0
        return 0.0, score_lower_is_better(after_noise - noise_floor, noise_floor)
    if after_noise <= noise_floor:
        return target_db, 100.0

    reduction_db = 20.0 * math.log10(before_noise / after_noise)
    score = float(np.clip(100.0 * reduction_db / target_db, 0.0, 100.0))
    return reduction_db, score


def score_blur_ratio(before_edge: float, after_edge: float, min_ratio: float) -> tuple[float, float]:
    """エッジ強度比とブラースコアを返す。"""
    if before_edge <= 0.0 or after_edge < 0.0 or not np.isfinite(before_edge + after_edge):
        return float("nan"), float("nan")
    ratio = after_edge / before_edge
    score = float(np.clip(100.0 * (ratio - min_ratio) / (1.0 - min_ratio), 0.0, 100.0))
    return float(ratio), score


def score_lower_is_better(value: float, reference: float) -> float:
    """小さいほど良い値を0..100点へ変換する。"""
    if value < 0.0 or not np.isfinite(value):
        return float("nan")
    return float(np.clip(100.0 * (1.0 - value / reference), 0.0, 100.0))


def validate_triplet(
    before: np.ndarray,
    after: np.ndarray,
    gt: np.ndarray,
    before_path: Path,
    after_path: Path,
    gt_path: Path,
) -> None:
    """before/after/GTフレームのshape/dtypeが一致していることを確認する。"""
    if before.shape != after.shape or before.shape != gt.shape:
        raise RuntimeError(
            "shape mismatch: "
            f"{before_path}={before.shape}, {after_path}={after.shape}, {gt_path}={gt.shape}"
        )
    if before.dtype != after.dtype or before.dtype != gt.dtype:
        raise RuntimeError(
            "dtype mismatch: "
            f"{before_path}={before.dtype}, {after_path}={after.dtype}, {gt_path}={gt.dtype}"
        )


def weighted_total(bands: dict[str, dict[str, Any]], weights: dict[str, float]) -> float:
    """各band/metricスコアの重み付き合計点を返す。"""
    weighted_sum = 0.0
    total_weight = 0.0
    for band_name, metrics in bands.items():
        for metric_name in ("noise", "blur", "motion"):
            score = metrics[f"{metric_name}_score"]
            weight = weights[f"{band_name}_{metric_name}"]
            if weight > 0.0 and np.isfinite(score):
                weighted_sum += score * weight
                total_weight += weight
    return weighted_sum / total_weight if total_weight > 0.0 else float("nan")


def evaluate_sequences(
    before_files: list[Path],
    after_files: list[Path],
    gt_files: list[Path],
    target_noise_db: float,
    blur_min_ratio: float,
    motion_error_ref: float,
    motion_threshold: float,
    noise_floor: float,
    min_pixels: int,
    weights: dict[str, float],
    dark_rois: list[Roi],
) -> dict[str, Any]:
    """before/after/GTの画像連番を比較して評価結果を返す。"""
    frame_count = min(len(before_files), len(after_files), len(gt_files))
    if frame_count < 2:
        raise RuntimeError("at least two frames are required for sequence evaluation")

    accum: dict[str, dict[str, list[float] | list[int]]] = {
        band.name: {
            "before_noise": [],
            "after_noise": [],
            "gt_edge": [],
            "after_edge": [],
            "motion_error": [],
            "noise_pixels": [],
            "edge_pixels": [],
            "motion_pixels": [],
        }
        for band in BANDS
    }

    first_before = read_image(before_files[0])
    first_after = read_image(after_files[0])
    first_gt = read_image(gt_files[0])
    validate_triplet(first_before, first_after, first_gt, before_files[0], after_files[0], gt_files[0])
    prev_gt_luma = to_luminance01(first_gt)

    shape = list(first_before.shape)
    dtype = str(first_before.dtype)
    validate_rois(dark_rois, shape)
    dark_roi_mask = make_roi_mask(prev_gt_luma.shape, dark_rois) if dark_rois else None

    for frame_index in range(frame_count):
        before = read_image(before_files[frame_index])
        after = read_image(after_files[frame_index])
        gt = read_image(gt_files[frame_index])
        validate_triplet(before, after, gt, before_files[frame_index], after_files[frame_index], gt_files[frame_index])

        before_luma = to_luminance01(before)
        after_luma = to_luminance01(after)
        gt_luma = to_luminance01(gt)
        gt_motion_luma = cv2.GaussianBlur(gt_luma, (9, 9), 0)
        prev_gt_motion_luma = cv2.GaussianBlur(prev_gt_luma, (9, 9), 0)
        motion_mask = np.abs(gt_motion_luma - prev_gt_motion_luma) > motion_threshold
        band_luma = np.maximum(gt_luma, prev_gt_luma)
        before_residual = before_luma - gt_luma
        after_residual = after_luma - gt_luma

        for band in BANDS:
            band_mask = make_band_mask(gt_luma, band)
            noise_base_mask = dark_roi_mask if band.name == "dark" and dark_roi_mask is not None else band_mask
            static_mask = erode_mask(noise_base_mask & ~motion_mask, 2)
            if int(np.count_nonzero(static_mask)) >= min_pixels:
                before_noise = float(np.std(before_residual[static_mask]))
                after_noise = float(np.std(after_residual[static_mask]))
                accum[band.name]["before_noise"].append(before_noise)
                accum[band.name]["after_noise"].append(after_noise)
                accum[band.name]["noise_pixels"].append(int(np.count_nonzero(static_mask)))

            gt_edge, edge_pixels = edge_strength(gt_luma, band_mask, min_pixels)
            after_edge, _ = edge_strength(after_luma, band_mask, min_pixels)
            if np.isfinite(gt_edge) and np.isfinite(after_edge):
                accum[band.name]["gt_edge"].append(gt_edge)
                accum[band.name]["after_edge"].append(after_edge)
                accum[band.name]["edge_pixels"].append(edge_pixels)

            motion_band_mask = motion_mask & make_band_mask(band_luma, band)
            if int(np.count_nonzero(motion_band_mask)) >= min_pixels:
                motion_error = float(np.mean(np.abs(after_residual)[motion_band_mask]))
                accum[band.name]["motion_error"].append(motion_error)
                accum[band.name]["motion_pixels"].append(int(np.count_nonzero(motion_band_mask)))

        prev_gt_luma = gt_luma

    bands: dict[str, dict[str, Any]] = {}
    for band in BANDS:
        values = accum[band.name]
        before_noise = safe_mean(values["before_noise"])  # type: ignore[arg-type]
        after_noise = safe_mean(values["after_noise"])  # type: ignore[arg-type]
        noise_reduction_db, noise_score = score_noise_reduction(
            before_noise,
            after_noise,
            target_noise_db,
            noise_floor,
        )

        gt_edge = safe_mean(values["gt_edge"])  # type: ignore[arg-type]
        after_edge = safe_mean(values["after_edge"])  # type: ignore[arg-type]
        edge_ratio, blur_score = score_blur_ratio(gt_edge, after_edge, blur_min_ratio)

        motion_error = safe_mean(values["motion_error"])  # type: ignore[arg-type]
        motion_score = score_lower_is_better(motion_error, motion_error_ref)

        bands[band.name] = {
            "before_noise": before_noise,
            "after_noise": after_noise,
            "noise_reduction_db": noise_reduction_db,
            "noise_score": noise_score,
            "gt_edge_strength": gt_edge,
            "after_edge_strength": after_edge,
            "edge_strength_ratio": edge_ratio,
            "blur_score": blur_score,
            "motion_error": motion_error,
            "motion_score": motion_score,
            "noise_pixels_mean": safe_mean(values["noise_pixels"]),  # type: ignore[arg-type]
            "edge_pixels_mean": safe_mean(values["edge_pixels"]),  # type: ignore[arg-type]
            "motion_pixels_mean": safe_mean(values["motion_pixels"]),  # type: ignore[arg-type]
        }

    return {
        "before_dir": str(before_files[0].parent),
        "after_dir": str(after_files[0].parent),
        "gt_dir": str(gt_files[0].parent),
        "frames": frame_count,
        "dtype": dtype,
        "shape": shape,
        "target_noise_db": target_noise_db,
        "blur_min_ratio": blur_min_ratio,
        "motion_error_ref": motion_error_ref,
        "motion_threshold": motion_threshold,
        "noise_floor": noise_floor,
        "dark_rois": [
            {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}
            for roi in dark_rois
        ],
        "score_weights": weights,
        "bands": bands,
        "total_score": weighted_total(bands, weights),
    }


def parse_weight(value: str) -> tuple[str, float]:
    """name=value形式の重み指定をパースする。"""
    if "=" not in value:
        raise argparse.ArgumentTypeError("use name=value")
    name, raw_weight = value.split("=", 1)
    name = name.strip()
    if name not in DEFAULT_SCORE_WEIGHTS:
        choices = ", ".join(sorted(DEFAULT_SCORE_WEIGHTS))
        raise argparse.ArgumentTypeError(f"unknown weight name: {name}; choices: {choices}")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be a number") from exc
    if weight < 0.0:
        raise argparse.ArgumentTypeError("weight must be >= 0")
    return name, weight


def parse_roi(value: str) -> Roi:
    """x,y,width,height形式のROI指定をパースする。"""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("use x,y,width,height")
    try:
        x, y, width, height = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI must be non-negative x/y and positive width/height")
    return Roi(x, y, width, height)


def weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    """デフォルト重みにCLI指定を反映する。"""
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    for name, weight in args.weight:
        weights[name] = weight
    return weights


def print_text_result(result: dict[str, Any]) -> None:
    """人が読みやすい形式で評価結果を表示する。"""
    print(f"before: {result['before_dir']}")
    print(f"after:  {result['after_dir']}")
    print(f"gt:     {result['gt_dir']}")
    print(f"frames: {result['frames']}, dtype: {result['dtype']}, shape: {result['shape']}")
    print(f"weights: {result['score_weights']}")
    if result["dark_rois"]:
        print(f"dark_rois: {result['dark_rois']}")
    print()
    print(
        "band,"
        "noise_score,noise_reduction_db,before_noise,after_noise,"
        "blur_score,edge_ratio,gt_edge,after_edge,"
        "motion_score,motion_error"
    )
    for band_name, metrics in result["bands"].items():
        print(
            f"{band_name},"
            f"{metrics['noise_score']:.2f},"
            f"{metrics['noise_reduction_db']:.3f},"
            f"{metrics['before_noise']:.6f},"
            f"{metrics['after_noise']:.6f},"
            f"{metrics['blur_score']:.2f},"
            f"{metrics['edge_strength_ratio']:.6f},"
            f"{metrics['gt_edge_strength']:.6f},"
            f"{metrics['after_edge_strength']:.6f},"
            f"{metrics['motion_score']:.2f},"
            f"{metrics['motion_error']:.6f}"
        )
    print()
    print(f"total_score: {result['total_score']:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare before/after/GT TIFF image sequences for noise, blur, and motion error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("before", type=Path, help="before-filter noisy TIFF image directory")
    parser.add_argument("after", type=Path, help="after-filter TIFF image directory")
    parser.add_argument("gt", type=Path, help="ground-truth TIFF image directory")
    parser.add_argument(
        "--target-noise-db",
        type=float,
        default=12.0,
        help="noise reduction dB that maps to 100 noise points",
    )
    parser.add_argument(
        "--blur-min-ratio",
        type=float,
        default=0.6,
        help="after/GT edge ratio that maps to 0 blur points",
    )
    parser.add_argument(
        "--motion-error-ref",
        type=float,
        default=0.08,
        help="normalized motion error that maps to 0 motion points",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.05,
        help="normalized GT-frame difference used to detect motion",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=DEFAULT_NOISE_FLOOR,
        help="normalized temporal noise treated as already clean",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=100,
        help="minimum pixels required for each band metric sample",
    )
    parser.add_argument(
        "--dark-roi",
        action="append",
        type=parse_roi,
        default=[],
        help="static dark ROI for dark-noise measurement as x,y,width,height; repeatable",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=parse_weight,
        default=[],
        help="override a total-score weight as name=value",
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main() -> int:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if not args.before.is_dir():
        parser.error(f"before must be a directory: {args.before}")
    if not args.after.is_dir():
        parser.error(f"after must be a directory: {args.after}")
    if not args.gt.is_dir():
        parser.error(f"gt must be a directory: {args.gt}")
    if args.target_noise_db <= 0:
        parser.error("target-noise-db must be positive")
    if not 0.0 <= args.blur_min_ratio < 1.0:
        parser.error("blur-min-ratio must be >= 0.0 and < 1.0")
    if args.motion_error_ref <= 0:
        parser.error("motion-error-ref must be positive")
    if args.motion_threshold <= 0:
        parser.error("motion-threshold must be positive")
    if args.noise_floor < 0:
        parser.error("noise-floor must be >= 0")
    if args.min_pixels <= 0:
        parser.error("min-pixels must be positive")

    try:
        before_files = list_image_files(args.before)
        after_files = list_image_files(args.after)
        gt_files = list_image_files(args.gt)
        if len(before_files) != len(after_files) or len(before_files) != len(gt_files):
            print(
                "warning: frame counts differ; "
                f"using {min(len(before_files), len(after_files), len(gt_files))} triplets "
                f"(before={len(before_files)}, after={len(after_files)}, gt={len(gt_files)})",
                file=sys.stderr,
            )
        weights = weights_from_args(args)
        if not any(weight > 0.0 for weight in weights.values()):
            parser.error("at least one weight must be > 0")

        result = evaluate_sequences(
            before_files,
            after_files,
            gt_files,
            args.target_noise_db,
            args.blur_min_ratio,
            args.motion_error_ref,
            args.motion_threshold,
            args.noise_floor,
            args.min_pixels,
            weights,
            args.dark_roi,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_text_result(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
