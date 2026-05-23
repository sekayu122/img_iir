#!/usr/bin/env python3
"""Dump image pixel values to CSV.

PNG/TIFFなどの画像をbit深度を保って読み込み、CSVに保存します。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_roi(value: str) -> tuple[int, int, int, int]:
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
    return x, y, width, height


def read_image(input_path: Path) -> np.ndarray:
    """画像を元のbit深度のまま読む。"""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read image: {input_path}")
    return image


def apply_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """必要ならROIで切り出す。"""
    if roi is None:
        return image

    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI exceeds image size: roi={x},{y},{width},{height}, "
            f"image={image_width}x{image_height}"
        )
    return image[y : y + height, x : x + width]


def select_channel(image: np.ndarray, channel: int | None) -> np.ndarray:
    """多チャンネル画像から1チャンネル、または全チャンネルを選ぶ。"""
    if image.ndim == 2:
        if channel is not None and channel != 0:
            raise ValueError("grayscale images only have channel 0")
        return image

    if channel is None:
        return image.reshape(image.shape[0], image.shape[1] * image.shape[2])

    if channel < 0 or channel >= image.shape[2]:
        raise ValueError(f"channel must be 0..{image.shape[2] - 1}")
    return image[:, :, channel]


def write_csv(output_path: Path, data: np.ndarray, delimiter: str) -> None:
    """画素値をCSVとして保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, fmt="%d", delimiter=delimiter)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump image pixel values to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument("output", type=Path, help="output CSV path")
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="channel index to dump; omit to dump all channels side by side",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="dump only x,y,width,height region",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter",
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

    try:
        image = read_image(args.input)
        cropped = apply_roi(image, args.roi)
        data = select_channel(cropped, args.channel)
        write_csv(args.output, data, args.delimiter)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.output} "
        f"(input={args.input}, dtype={image.dtype}, shape={image.shape}, "
        f"dump_shape={data.shape})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
