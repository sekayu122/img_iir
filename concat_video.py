#!/usr/bin/env python3
"""Place two videos or image sequences side by side.

OpenCVで2本の動画または2つの画像フォルダを読み込み、左・右に並べた
動画または画像連番として書き出します。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def open_capture(input_path: Path) -> cv2.VideoCapture:
    """入力動画を開く。"""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")
    return capture


def read_video_info(capture: cv2.VideoCapture) -> tuple[int, int, float, int]:
    """入力動画の幅、高さ、FPS、フレーム数を取得する。"""
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    if width <= 0 or height <= 0:
        raise RuntimeError("could not read input video size")
    if fps <= 0:
        fps = 30.0

    return width, height, fps, frame_count


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    """アスペクト比を保って指定高さへリサイズする。"""
    height, width = frame.shape[:2]
    if height == target_height:
        return frame

    target_width = max(1, int(round(width * target_height / height)))
    interpolation = cv2.INTER_AREA if target_height < height else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def frame_to_rgb8(frame: np.ndarray) -> np.ndarray:
    """動画書き出し用にRGB uint8へ変換する。"""
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)

    if frame.dtype == np.uint8:
        scaled = frame
    elif frame.dtype == np.uint16:
        scaled = np.rint(frame.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
    else:
        min_value = float(np.min(frame))
        max_value = float(np.max(frame))
        if max_value <= min_value:
            scaled = np.zeros(frame.shape, dtype=np.uint8)
        else:
            scaled = np.rint((frame.astype(np.float32) - min_value) * (255.0 / (max_value - min_value)))
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    if scaled.shape[2] == 4:
        return cv2.cvtColor(scaled, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)


def iter_side_by_side_frames(
    left_capture: cv2.VideoCapture,
    right_capture: cv2.VideoCapture,
    target_height: int,
    max_frames: int | None,
) -> Iterator[np.ndarray]:
    """左右に並べたRGBフレームを順に返す。"""
    written = 0
    while max_frames is None or written < max_frames:
        left_ok, left_frame = left_capture.read()
        right_ok, right_frame = right_capture.read()
        if not left_ok or not right_ok:
            break

        left_resized = resize_to_height(left_frame, target_height)
        right_resized = resize_to_height(right_frame, target_height)
        yield frame_to_rgb8(np.hstack((left_resized, right_resized)))
        written += 1


def list_image_files(input_dir: Path) -> list[Path]:
    """画像フォルダ内の画像ファイルを名前順に返す。"""
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise RuntimeError(f"no image files found: {input_dir}")
    return files


def read_image(input_path: Path) -> np.ndarray:
    """画像を元のbit深度のまま読む。"""
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read image: {input_path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image


def iter_side_by_side_images(
    left_files: list[Path],
    right_files: list[Path],
    target_height: int,
) -> Iterator[np.ndarray]:
    """画像連番を左右に並べたBGR/BGRAフレームとして順に返す。"""
    for left_path, right_path in zip(left_files, right_files):
        left_frame = resize_to_height(read_image(left_path), target_height)
        right_frame = resize_to_height(read_image(right_path), target_height)
        if left_frame.dtype != right_frame.dtype:
            raise RuntimeError(
                f"image dtype mismatch: {left_path} is {left_frame.dtype}, "
                f"{right_path} is {right_frame.dtype}"
            )
        if left_frame.shape[2] != right_frame.shape[2]:
            raise RuntimeError(
                f"image channel mismatch: {left_path} has {left_frame.shape[2]}, "
                f"{right_path} has {right_frame.shape[2]}"
            )
        yield np.hstack((left_frame, right_frame))


def iter_rgb8_from_bgr_frames(frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    """BGR/BGRAフレーム列を動画用RGB uint8に変換する。"""
    for frame in frames:
        yield frame_to_rgb8(frame)


def write_video(output: Path, fps: float, crf: int, frames: Iterator[np.ndarray]) -> int:
    """imageio + ffmpegでRGBフレーム列をH.264 MP4として保存する。"""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required. Install it with: pip install imageio imageio-ffmpeg"
        ) from exc

    frame_count = 0
    with imageio.get_writer(
        str(output),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", str(crf), "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
            frame_count += 1

    return frame_count


def write_image_sequence(output_dir: Path, frames: Iterator[np.ndarray], extension: str) -> int:
    """左右結合済みフレームを画像連番として保存する。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_count = 0
    try:
        for frame_index, frame in enumerate(frames):
            output_path = output_dir / f"frame_{frame_index:04d}.{extension}"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"could not write image: {output_path}")
            frame_count += 1
    except Exception:
        if frame_count == 0:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return frame_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place two videos or image sequences side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("left", type=Path, help="left input video path or image directory")
    parser.add_argument("right", type=Path, help="right input video path or image directory")
    parser.add_argument("output", type=Path, help="output MP4 path or image directory")
    parser.add_argument(
        "--mode",
        choices=("auto", "video", "images"),
        default="auto",
        help="input mode; auto uses directory inputs as image sequences",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="output frame height; default uses the smaller input height",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output fps; default uses the left input fps",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg H.264 quality; lower is higher quality",
    )
    parser.add_argument(
        "--image-extension",
        choices=("tiff", "png", "bmp"),
        default="tiff",
        help="extension for image-sequence output when output is a directory",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    """入力モードを決める。"""
    if args.mode != "auto":
        return args.mode
    if args.left.is_dir() and args.right.is_dir():
        return "images"
    return "video"


def should_write_video(output: Path) -> bool:
    """出力先の拡張子から動画出力か画像フォルダ出力かを判定する。"""
    return output.suffix.lower() in VIDEO_SUFFIXES


def main() -> int:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if not args.left.exists():
        parser.error(f"left input not found: {args.left}")
    if not args.right.exists():
        parser.error(f"right input not found: {args.right}")
    if args.height is not None and args.height <= 0:
        parser.error("height must be positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("fps must be positive")
    if args.crf < 0 or args.crf > 51:
        parser.error("crf must be 0..51")

    mode = resolve_mode(args)
    if mode == "video" and (args.left.is_dir() or args.right.is_dir()):
        parser.error("video mode requires two video files")
    if mode == "images" and (not args.left.is_dir() or not args.right.is_dir()):
        parser.error("images mode requires two image directories")

    if should_write_video(args.output):
        args.output.parent.mkdir(parents=True, exist_ok=True)
    elif args.output.exists():
        parser.error(f"image output directory already exists: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    if mode == "video":
        left_capture = open_capture(args.left)
        right_capture = open_capture(args.right)
        try:
            left_width, left_height, left_fps, left_frames = read_video_info(left_capture)
            right_width, right_height, right_fps, right_frames = read_video_info(right_capture)

            output_height = args.height or min(left_height, right_height)
            output_fps = args.fps or left_fps
            max_frames = None
            if left_frames > 0 and right_frames > 0:
                max_frames = min(left_frames, right_frames)

            frames = iter_side_by_side_frames(
                left_capture,
                right_capture,
                output_height,
                max_frames,
            )
            output_frame_count = write_video(args.output, output_fps, args.crf, frames)
        finally:
            left_capture.release()
            right_capture.release()

        print(
            f"wrote {args.output} "
            f"(mode=video, left={left_width}x{left_height}@{left_fps:.3f}, "
            f"right={right_width}x{right_height}@{right_fps:.3f}, "
            f"height={output_height}, fps={output_fps:.3f}, "
            f"frames={output_frame_count})"
        )
    else:
        left_files = list_image_files(args.left)
        right_files = list_image_files(args.right)
        first_left = read_image(left_files[0])
        first_right = read_image(right_files[0])
        output_height = args.height or min(first_left.shape[0], first_right.shape[0])
        output_fps = args.fps or 30.0
        source_frames = iter_side_by_side_images(left_files, right_files, output_height)

        if should_write_video(args.output):
            output_frame_count = write_video(
                args.output,
                output_fps,
                args.crf,
                iter_rgb8_from_bgr_frames(source_frames),
            )
            output_kind = "video"
        else:
            output_frame_count = write_image_sequence(
                args.output,
                source_frames,
                args.image_extension,
            )
            output_kind = "images"

        print(
            f"wrote {args.output} "
            f"(mode=images, output={output_kind}, "
            f"left_images={len(left_files)}, right_images={len(right_files)}, "
            f"height={output_height}, fps={output_fps:.3f}, "
            f"frames={output_frame_count})"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
