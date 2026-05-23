#!/usr/bin/env python3
"""Generate dark noisy MP4 test videos or image sequences with luminance squares.

The frame content is intentionally simple: a dark background, five fixed
rectangles at configurable luminance levels, and temporally changing noise.
NumPyでフレームを作り、MP4またはTIFF連番を書き出します。

暗い背景の上に輝度の違う四角を並べ、粒度と強さを指定したノイズを重ねて
フィルタ検討用のMP4動画を生成します。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


DEFAULT_BRIGHTNESSES = "240,60,250,100,50,40,255"


def parse_int_list(value: str) -> list[int]:
    """コマンドラインの "24,48,80" のような指定を整数リストに変換する。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one integer is required")
    try:
        values = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc
    for item in values:
        if item < 0 or item > 255:
            raise argparse.ArgumentTypeError("luminance values must be 0..255")
    return values


def max_value_for_bit_depth(bit_depth: int) -> int:
    """bit幅から最大コード値を返す。"""
    return (1 << bit_depth) - 1


def scale_8bit_value(value: float, bit_depth: int) -> float:
    """既存CLIの0..255指定を指定bit幅のコード値へ変換する。"""
    return value * max_value_for_bit_depth(bit_depth) / 255.0


def quantize_frame(frame: np.ndarray, bit_depth: int, storage_dtype: np.dtype) -> np.ndarray:
    """指定bit幅の範囲に丸めて保存用dtypeへ変換する。"""
    max_value = max_value_for_bit_depth(bit_depth)
    return np.clip(np.rint(frame), 0, max_value).astype(storage_dtype)


def make_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    strength: float,
    grain_size: int,
    mode: str,
) -> np.ndarray:
    """指定した粒度と強さで1フレーム分の輝度ノイズを作る。"""
    if strength <= 0:
        return np.zeros((height, width), dtype=np.float32)

    # grain_size単位の小さいノイズ画像を作り、拡大してブロック状の粒度を表現する。
    grain_size = max(1, int(grain_size))
    small_h = (height + grain_size - 1) // grain_size
    small_w = (width + grain_size - 1) // grain_size

    # gaussianは標準偏差、uniformは最大振れ幅としてnoise-strengthを使う。
    if mode == "gaussian":
        small = rng.normal(0.0, strength, size=(small_h, small_w))
    else:
        small = rng.uniform(-strength, strength, size=(small_h, small_w))

    noise = np.repeat(np.repeat(small, grain_size, axis=0), grain_size, axis=1)
    return noise[:height, :width].astype(np.float32)


def square_rects(
    width: int,
    height: int,
    count: int,
    square_size: int | None,
    margin: int,
) -> list[tuple[int, int, int, int]]:
    """四角を横一列に中央配置するための座標リストを作る。"""
    if square_size is None:
        # サイズ未指定時は、指定個数の四角が余白込みで画面に収まる最大寄りの大きさにする。
        usable_w = width - margin * (count + 1)
        usable_h = height - margin * 2
        square_size = max(8, min(usable_h, usable_w // count))

    total_w = count * square_size + (count - 1) * margin
    x0 = max(0, (width - total_w) // 2)
    y0 = max(0, (height - square_size) // 2)

    rects = []
    for idx in range(count):
        x1 = x0 + idx * (square_size + margin)
        y1 = y0
        x2 = min(width, x1 + square_size)
        y2 = min(height, y1 + square_size)
        rects.append((x1, y1, x2, y2))
    return rects


def reflect_position(start: int, velocity: float, size: int, limit: int, time_sec: float) -> int:
    """画面端で反射する1次元位置を計算する。"""
    span = max(0, limit - size)
    if span == 0:
        return 0

    # 0..span..0の三角波にして、端で跳ね返る動きを作る。
    raw = (float(start) + velocity * time_sec) % (span * 2)
    if raw > span:
        raw = span * 2 - raw
    return int(round(raw))


def motion_directions(count: int) -> list[tuple[float, float]]:
    """各四角に割り当てる移動方向を作る。横・縦・斜めを混ぜる。"""
    base = [
        (1.0, 0.0),   # 横
        (0.0, 1.0),   # 縦
        (1.0, 1.0),   # 斜め
        (-1.0, 0.0),  # 横の逆方向
        (1.0, -1.0),  # 斜めの別方向
        (-1.0, 1.0),
        (0.0, -1.0),
        (-1.0, -1.0),
    ]

    directions = []
    for idx in range(count):
        dx, dy = base[idx % len(base)]
        length = (dx * dx + dy * dy) ** 0.5
        directions.append((dx / length, dy / length))
    return directions


def moving_square_rects(
    initial_rects: list[tuple[int, int, int, int]],
    brightnesses: list[int],
    width: int,
    height: int,
    frame_index: int,
    fps: int,
    max_speed: float,
) -> list[tuple[int, int, int, int]]:
    """フレーム番号に応じて、移動後の四角座標を計算する。"""
    time_sec = frame_index / fps
    directions = motion_directions(len(initial_rects))
    rects = []
    last_index = max(1, len(initial_rects) - 1)
    for index, ((x1, y1, x2, y2), (dx, dy)) in enumerate(zip(
        initial_rects,
        directions,
    )):
        square_w = x2 - x1
        square_h = y2 - y1
        # 箱の並び順で徐々に速くする。左端は停止、右端はmax_speed。
        speed_ratio = index / last_index
        speed = max_speed * speed_ratio
        moved_x = reflect_position(x1, dx * speed, square_w, width, time_sec)
        moved_y = reflect_position(y1, dy * speed, square_h, height, time_sec)
        rects.append((moved_x, moved_y, moved_x + square_w, moved_y + square_h))
    return rects


def make_frame(
    width: int,
    height: int,
    background: float,
    brightnesses: list[int],
    rects: list[tuple[int, int, int, int]],
    noise: np.ndarray,
    clean_squares: bool,
    bit_depth: int,
    storage_dtype: np.dtype,
) -> np.ndarray:
    """背景、輝度四角、ノイズを合成してRGBフレームを作る。"""
    frame = np.full((height, width), float(background), dtype=np.float32)
    for brightness, (x1, y1, x2, y2) in zip(brightnesses, rects):
        frame[y1:y2, x1:x2] = scale_8bit_value(brightness, bit_depth)

    if clean_squares:
        # clean_squares指定時は、背景にはノイズを乗せつつ四角の輝度は固定値に戻す。
        noisy = frame + noise
        for brightness, (x1, y1, x2, y2) in zip(brightnesses, rects):
            noisy[y1:y2, x1:x2] = scale_8bit_value(brightness, bit_depth)
    else:
        noisy = frame + noise

    gray = quantize_frame(noisy, bit_depth, storage_dtype)
    return np.repeat(gray[:, :, None], 3, axis=2)


def iter_frames(
    args: argparse.Namespace,
    frame_count: int,
    storage_dtype: np.dtype,
) -> Iterator[np.ndarray]:
    """設定値から動画フレームを1枚ずつ生成する。"""
    initial_rects = square_rects(
        args.width,
        args.height,
        len(args.brightnesses),
        args.square_size,
        args.margin,
    )
    rng = np.random.default_rng(args.seed)
    previous_noise = np.zeros((args.height, args.width), dtype=np.float32)
    alpha = float(args.temporal_correlation)

    for frame_index in range(frame_count):
        # temporal_correlationを大きくすると、前フレームのノイズが残ってちらつきが遅くなる。
        fresh_noise = make_noise(
            rng,
            args.height,
            args.width,
            scale_8bit_value(args.noise_strength, args.bit_depth),
            args.grain_size,
            args.noise_mode,
        )
        noise = alpha * previous_noise + (1.0 - alpha) * fresh_noise
        previous_noise = noise

        # 四角はそれぞれ横・縦・斜め方向へ動き、画面端で反射する。
        rects = moving_square_rects(
            initial_rects,
            args.brightnesses,
            args.width,
            args.height,
            frame_index,
            args.fps,
            args.motion_max_speed,
        )

        yield make_frame(
            args.width,
            args.height,
            scale_8bit_value(args.background, args.bit_depth),
            args.brightnesses,
            rects,
            noise,
            args.clean_squares,
            args.bit_depth,
            storage_dtype,
        )


def write_video(output: Path, fps: int, crf: int, frames: Iterator[np.ndarray]) -> None:
    """imageio + ffmpegでRGBフレーム列をH.264 MP4として保存する。"""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required. Install it with: pip install imageio imageio-ffmpeg"
        ) from exc

    with imageio.get_writer(
        str(output),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", str(crf)],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def write_image_sequence(
    output_dir: Path,
    image_extension: str,
    frames: Iterator[np.ndarray],
) -> int:
    """TIFF連番を新しいフォルダへ保存する。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_count = 0
    try:
        for frame_index, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{frame_index:04d}.{image_extension}"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(frame_path), bgr):
                raise RuntimeError(f"could not write image: {frame_path}")
            frame_count += 1
    except Exception:
        if frame_count == 0:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    return frame_count


def output_storage_dtype(output_format: str) -> np.dtype:
    """出力形式に対応するNumPy dtypeを返す。"""
    if output_format == "mp4":
        return np.dtype(np.uint8)
    if output_format == "tiff16":
        return np.dtype(np.uint16)
    raise ValueError(f"unsupported output format: {output_format}")


def image_extension_for_output_format(output_format: str) -> str:
    """連番画像の拡張子を返す。"""
    if output_format == "tiff16":
        return "tiff"
    raise ValueError(f"not an image output format: {output_format}")


def build_arg_parser() -> argparse.ArgumentParser:
    """動画生成パラメータをコマンドラインから受け取る設定。"""
    parser = argparse.ArgumentParser(
        description="Generate dark noisy test data with luminance squares.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output", type=Path, help="output MP4 path or image sequence directory")
    parser.add_argument("--width", type=int, default=800, help="video width")
    parser.add_argument("--height", type=int, default=800, help="video height")
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="number of frames to generate; overrides duration when specified",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=16,
        help="background luminance, 0 is black and 255 is white",
    )
    parser.add_argument(
        "--brightnesses",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_BRIGHTNESSES),
        help=f"comma-separated square luminances; default {DEFAULT_BRIGHTNESSES}",
    )
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=18.0,
        help="noise amplitude/stddev in luminance levels",
    )
    parser.add_argument(
        "--grain-size",
        type=int,
        default=2,
        help="noise block size in pixels; 1 is fine pixel noise",
    )
    parser.add_argument(
        "--noise-mode",
        choices=("gaussian", "uniform"),
        default="gaussian",
        help="noise distribution",
    )
    parser.add_argument(
        "--temporal-correlation",
        type=float,
        default=0.0,
        help="0.0 is independent noise per frame; values near 1.0 persist longer",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=None,
        help="square side length in pixels; auto when omitted",
    )
    parser.add_argument(
        "--motion-max-speed",
        "--motion-speed",
        dest="motion_max_speed",
        type=float,
        default=500.0,
        help="maximum square movement speed in pixels per second; rightmost square uses this speed and leftmost square is stopped",
    )
    parser.add_argument("--margin", type=int, default=16, help="spacing between squares")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--clean-squares",
        action="store_true",
        help="apply noise only to the background, not inside the squares",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg H.264 quality; lower is higher quality",
    )
    parser.add_argument(
        "--output-format",
        choices=("mp4", "tiff16"),
        default="mp4",
        help="output format; tiff16 writes one file per frame into a new directory",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=8,
        help="generated sample bit depth",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """不正な値を早めに検出して、生成途中の失敗を避ける。"""
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.frames is not None and args.frames <= 0:
        raise ValueError("frames must be positive")
    if args.background < 0 or args.background > 255:
        raise ValueError("background must be 0..255")
    if args.grain_size <= 0:
        raise ValueError("grain-size must be positive")
    if args.square_size is not None and args.square_size <= 0:
        raise ValueError("square-size must be positive")
    if args.motion_max_speed < 0:
        raise ValueError("motion-max-speed must be >= 0")
    if not 0.0 <= args.temporal_correlation < 1.0:
        raise ValueError("temporal-correlation must be >= 0.0 and < 1.0")

    compatible_bit_depths = {
        "mp4": {8},
        "tiff16": {16},
    }
    allowed = compatible_bit_depths[args.output_format]
    if args.bit_depth not in allowed:
        allowed_text = ", ".join(f"{bit_depth}bit" for bit_depth in sorted(allowed))
        raise ValueError(
            f"{args.output_format} supports {allowed_text}; "
            f"got {args.bit_depth}bit"
        )
    if args.output_format != "mp4" and args.output.exists():
        raise ValueError(f"image output directory already exists: {args.output}")


def main() -> int:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    # 動画全体で使う固定情報を先に計算する。
    frame_count = args.frames or max(1, int(round(args.duration * args.fps)))
    storage_dtype = output_storage_dtype(args.output_format)
    # 1フレームずつ生成してすぐ書き出すので、長い動画でもメモリを使いすぎない。
    frames = iter_frames(args, frame_count, storage_dtype)
    if args.output_format == "mp4":
        write_video(output, args.fps, args.crf, frames)
        output_count = frame_count
    else:
        output_count = write_image_sequence(
            output,
            image_extension_for_output_format(args.output_format),
            frames,
        )

    print(
        f"wrote {output} "
        f"({args.width}x{args.height}, {output_count} frames, "
        f"format={args.output_format}, bit_depth={args.bit_depth})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
