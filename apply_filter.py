#!/usr/bin/env python3
"""Apply an IIR filter to a TIFF image sequence.

入力フォルダ内の連番TIFFを読み込み、フィルタ結果をtiff16連番として
新しい出力フォルダへ保存します。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from iir_filters import create_filter


IMAGE_SUFFIXES = {".tif", ".tiff"}


def list_image_files(input_dir: Path) -> list[Path]:
    """入力フォルダ内のTIFF画像を名前順に返す。"""
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise RuntimeError(f"no TIFF images found: {input_dir}")
    return files


def read_image(input_path: Path) -> np.ndarray:
    """TIFF画像をbit深度を保って読む。"""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read image: {input_path}")
    if image.dtype != np.uint16:
        raise RuntimeError(f"expected uint16 TIFF, got {image.dtype}: {input_path}")
    return image


def validate_same_shape(reference: np.ndarray, image: np.ndarray, input_path: Path) -> None:
    """全フレームのshapeとdtypeが同じであることを確認する。"""
    if image.shape != reference.shape:
        raise RuntimeError(
            f"image shape mismatch: first={reference.shape}, "
            f"{input_path}={image.shape}"
        )
    if image.dtype != reference.dtype:
        raise RuntimeError(
            f"image dtype mismatch: first={reference.dtype}, "
            f"{input_path}={image.dtype}"
        )


def quantize_tiff16(frame: np.ndarray) -> np.ndarray:
    """フィルタ結果をtiff16保存用に丸める。"""
    return np.clip(np.rint(frame), 0, 65535).astype(np.uint16)


def apply_filter_to_sequence(
    input_files: list[Path],
    output_dir: Path,
    filter_name: str,
    alpha: float,
) -> int:
    """連番画像にフィルタを適用して出力フォルダへ保存する。"""
    frame_filter = create_filter(filter_name, alpha)
    output_dir.mkdir(parents=True, exist_ok=False)

    output_count = 0
    reference: np.ndarray | None = None
    try:
        for frame_index, input_path in enumerate(input_files):
            image = read_image(input_path)
            if reference is None:
                reference = image
            else:
                validate_same_shape(reference, image, input_path)

            filtered = quantize_tiff16(frame_filter.apply(image))
            output_path = output_dir / f"frame_{frame_index:04d}.tiff"
            if not cv2.imwrite(str(output_path), filtered):
                raise RuntimeError(f"could not write image: {output_path}")
            output_count += 1
    except Exception:
        if output_count == 0:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    return output_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an IIR filter to a TIFF image sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input TIFF image directory")
    parser.add_argument("output", type=Path, help="new output TIFF image directory")
    parser.add_argument(
        "--filter",
        choices=("alpha",),
        default="alpha",
        help="filter algorithm",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="current-frame blend amount; output[n]=alpha*input[n]+(1-alpha)*output[n-1]",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    if not args.input.is_dir():
        parser.error(f"input must be a directory: {args.input}")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be 0.0..1.0")

    try:
        input_files = list_image_files(args.input)
        output_count = apply_filter_to_sequence(
            input_files,
            args.output,
            args.filter,
            args.alpha,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.output} "
        f"(input={args.input}, frames={output_count}, "
        f"filter={args.filter}, alpha={args.alpha})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
