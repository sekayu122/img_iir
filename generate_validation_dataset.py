#!/usr/bin/env python3
"""Generate a synthetic video-like TIFF16 validation dataset.

出力フォルダに `src`, `test`, `gt` を作り、run_experiment.py にそのまま渡せる
連続フレームデータを生成します。矩形中心のtrainデータとは違い、背景テクスチャ、
カメラ揺れ、複数の動体、輝度依存ノイズを含む実写風の合成データです。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from skimage import data as skimage_data


@dataclass(frozen=True)
class MovingObject:
    """合成シーン内を動く物体。"""

    kind: str
    center: tuple[float, float]
    size: tuple[float, float]
    velocity: tuple[float, float]
    color: tuple[float, float, float]


def max_value_for_bit_depth(bit_depth: int) -> int:
    return (1 << bit_depth) - 1


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def make_low_frequency_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    channels: int,
    scale: int,
) -> np.ndarray:
    small_h = max(2, height // scale)
    small_w = max(2, width // scale)
    small = rng.normal(0.0, 1.0, size=(small_h, small_w, channels)).astype(np.float32)
    resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    if channels == 1 and resized.ndim == 2:
        resized = resized[:, :, None]
    return resized.astype(np.float32)


SKIMAGE_IMAGE_LOADERS = {
    "astronaut": skimage_data.astronaut,
    "camera": skimage_data.camera,
    "chelsea": skimage_data.chelsea,
    "coffee": skimage_data.coffee,
    "rocket": skimage_data.rocket,
    "immunohistochemistry": skimage_data.immunohistochemistry,
    "hubble_deep_field": skimage_data.hubble_deep_field,
}


def skimage_image_choices() -> tuple[str, ...]:
    return ("auto", *SKIMAGE_IMAGE_LOADERS.keys())


def load_skimage_image(name: str, rng: np.random.Generator) -> np.ndarray:
    """scikit-imageのサンプル画像を0..1 BGRで読む。"""
    if name == "auto":
        names = tuple(SKIMAGE_IMAGE_LOADERS)
        name = names[int(rng.integers(0, len(names)))]
    image = SKIMAGE_IMAGE_LOADERS[name]()
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.max() > 1.0:
        image /= 255.0
    return image[:, :, ::-1].copy()


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """アスペクト比を保って指定サイズを覆うようにリサイズして中央クロップする。"""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized_w = max(width, int(round(src_w * scale)))
    resized_h = max(height, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].astype(np.float32)


def make_skimage_world(
    width: int,
    height: int,
    rng: np.random.Generator,
    image_name: str,
) -> np.ndarray:
    """scikit-imageのサンプル画像からvalidation用背景ワールドを作る。"""
    image = load_skimage_image(image_name, rng)
    world = resize_cover(image, width, height)

    # 単一画像だけだと輝度帯が偏ることがあるため、評価用の暗部/高輝度領域を薄く追加する。
    overlay = world.copy()
    cv2.rectangle(overlay, (32, 36), (max(96, width // 4), max(96, height // 4)), (0.045, 0.050, 0.060), -1)
    cv2.rectangle(overlay, (width - max(150, width // 5), height - max(120, height // 4)), (width - 36, height - 32), (0.90, 0.88, 0.78), -1)
    world = cv2.addWeighted(world, 0.82, overlay, 0.18, 0.0)

    texture = make_low_frequency_noise(rng, height, width, 3, 28) * 0.012
    return np.clip(world + texture, 0.0, 1.0).astype(np.float32)


def make_static_world(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """実写風の静止背景ワールドを0..1 BGRで作る。"""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)

    wall = 0.13 + 0.42 * xn + 0.16 * yn
    floor_mask = yn > 0.62
    floor = 0.20 + 0.28 * (1.0 - yn) + 0.08 * np.sin(xn * np.pi * 6.0)
    luma = np.where(floor_mask, floor, wall)

    texture = (
        0.030 * make_low_frequency_noise(rng, height, width, 1, 18)[:, :, 0]
        + 0.018 * make_low_frequency_noise(rng, height, width, 1, 47)[:, :, 0]
        + 0.010 * np.sin((x * 0.06 + y * 0.025))
    )
    luma = np.clip(luma + texture, 0.02, 0.94)

    # BGR各チャンネルに少し違う色味を持たせる。
    world = np.stack(
        [
            luma * (0.92 + 0.06 * yn),
            luma * (1.00 - 0.03 * xn),
            luma * (1.04 - 0.10 * yn),
        ],
        axis=2,
    ).astype(np.float32)

    # 暗部、通常輝度、高輝度の静止領域を意図的に配置する。
    cv2.rectangle(world, (40, 45), (width // 3, height // 3), (0.055, 0.060, 0.070), -1)
    cv2.rectangle(world, (width // 2, 80), (width - 70, height // 3 + 35), (0.46, 0.50, 0.55), -1)
    cv2.rectangle(world, (width - 190, height - 160), (width - 55, height - 40), (0.88, 0.86, 0.78), -1)

    # 細かい構造物。エッジ保持とブラー検出に効く。
    for _ in range(36):
        x1 = int(rng.integers(20, width - 80))
        y1 = int(rng.integers(25, height - 45))
        w = int(rng.integers(25, 120))
        h = int(rng.integers(8, 55))
        base = float(rng.uniform(0.08, 0.86))
        color = (base * rng.uniform(0.85, 1.05), base, base * rng.uniform(0.9, 1.15))
        cv2.rectangle(world, (x1, y1), (min(width - 1, x1 + w), min(height - 1, y1 + h)), color, 1)

    for _ in range(22):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        val = float(rng.uniform(0.10, 0.80))
        cv2.line(world, p1, p2, (val, val * 1.03, val * 0.96), 1, cv2.LINE_AA)

    return np.clip(world, 0.0, 1.0)


def reflect_position(position: float, velocity: float, size: float, limit: int, time_sec: float) -> float:
    span = max(1.0, float(limit) - size)
    raw = (position + velocity * time_sec) % (span * 2.0)
    return span * 2.0 - raw if raw > span else raw


def draw_moving_object(frame: np.ndarray, obj: MovingObject, frame_index: int, fps: float) -> None:
    time_sec = frame_index / fps
    height, width = frame.shape[:2]
    w, h = obj.size
    cx = reflect_position(obj.center[0], obj.velocity[0], w, width, time_sec) + w * 0.5
    cy = reflect_position(obj.center[1], obj.velocity[1], h, height, time_sec) + h * 0.5
    color = obj.color

    if obj.kind == "ellipse":
        cv2.ellipse(
            frame,
            (int(round(cx)), int(round(cy))),
            (int(round(w * 0.5)), int(round(h * 0.5))),
            12.0 * np.sin(time_sec * 1.4),
            0,
            360,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            frame,
            (int(round(cx)), int(round(cy))),
            (int(round(w * 0.25)), int(round(h * 0.25))),
            0,
            0,
            360,
            tuple(min(1.0, c + 0.18) for c in color),
            1,
            cv2.LINE_AA,
        )
    elif obj.kind == "box":
        x1 = int(round(cx - w * 0.5))
        y1 = int(round(cy - h * 0.5))
        x2 = int(round(cx + w * 0.5))
        y2 = int(round(cy + h * 0.5))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), tuple(min(1.0, c + 0.22) for c in color), 2, cv2.LINE_AA)
    else:
        p1 = (int(round(cx - w * 0.5)), int(round(cy - h * 0.5)))
        p2 = (int(round(cx + w * 0.5)), int(round(cy + h * 0.5)))
        cv2.line(frame, p1, p2, color, max(2, int(min(w, h) * 0.18)), cv2.LINE_AA)


def crop_with_camera_motion(world: np.ndarray, width: int, height: int, frame_index: int, fps: float) -> np.ndarray:
    """大きめのworldからカメラ揺れ付きでフレームを切り出す。"""
    margin_x = (world.shape[1] - width) * 0.5
    margin_y = (world.shape[0] - height) * 0.5
    t = frame_index / fps
    dx = margin_x + 7.0 * np.sin(2.0 * np.pi * 0.23 * t) + 2.0 * np.sin(2.0 * np.pi * 0.91 * t)
    dy = margin_y + 5.0 * np.cos(2.0 * np.pi * 0.19 * t) + 1.5 * np.sin(2.0 * np.pi * 0.67 * t)
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(
        world,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def make_clean_frame(
    world: np.ndarray,
    objects: list[MovingObject],
    width: int,
    height: int,
    frame_index: int,
    fps: float,
) -> np.ndarray:
    frame = crop_with_camera_motion(world, width, height, frame_index, fps)
    frame = frame.copy()

    # 緩やかな露光変動。IIRが明るさ変化に遅れないかを見る。
    t = frame_index / fps
    exposure = 1.0 + 0.025 * np.sin(2.0 * np.pi * 0.11 * t)
    frame *= exposure

    for obj in objects:
        draw_moving_object(frame, obj, frame_index, fps)

    # 小さな点光源。高輝度の残像チェック用。
    x = int(width * (0.72 + 0.12 * np.sin(2.0 * np.pi * 0.31 * t)))
    y = int(height * (0.22 + 0.08 * np.cos(2.0 * np.pi * 0.29 * t)))
    cv2.circle(frame, (x, y), max(3, width // 120), (0.96, 0.93, 0.76), -1, cv2.LINE_AA)
    return np.clip(frame, 0.0, 1.0)


def add_sensor_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    fixed_pattern: np.ndarray,
    previous_noise: np.ndarray,
    temporal_correlation: float,
    noise_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """輝度依存ノイズ、RGB独立ノイズ、固定パターンノイズを追加する。"""
    luma = 0.0722 * clean[:, :, 0] + 0.7152 * clean[:, :, 1] + 0.2126 * clean[:, :, 2]

    # 暗部の読み出しノイズと、明るさに応じたショットノイズを混ぜる。
    dark_weight = np.clip((0.45 - luma) / 0.45, 0.0, 1.0)
    sigma = noise_strength * (0.35 + 0.70 * np.sqrt(np.clip(luma, 0.0, 1.0)) + 0.75 * dark_weight)
    sigma = sigma[:, :, None]

    fresh = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * sigma
    chroma = rng.normal(0.0, noise_strength * 0.18, size=clean.shape).astype(np.float32)
    row_noise = rng.normal(0.0, noise_strength * 0.10, size=(clean.shape[0], 1, 1)).astype(np.float32)
    noise = temporal_correlation * previous_noise + (1.0 - temporal_correlation) * (fresh + chroma + row_noise)

    noisy = clean + noise + fixed_pattern
    return np.clip(noisy, 0.0, 1.0), noise.astype(np.float32)


def quantize_tiff16(frame01: np.ndarray, bit_depth: int) -> np.ndarray:
    max_value = max_value_for_bit_depth(bit_depth)
    storage_max = np.iinfo(np.uint16).max
    frame = np.clip(frame01, 0.0, 1.0) * max_value
    if bit_depth < 16:
        frame *= storage_max / max_value
    return np.clip(np.rint(frame), 0, storage_max).astype(np.uint16)


def write_tiff(path: Path, frame01: np.ndarray, bit_depth: int) -> None:
    output = quantize_tiff16(frame01, bit_depth)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"could not write image: {path}")


def video_frame_u8(frame01: np.ndarray) -> np.ndarray:
    """確認用MP4へ書くため、0..1 BGR frameを8bit RGBへ変換する。"""
    bgr = np.clip(np.rint(frame01 * 255.0), 0, 255).astype(np.uint8)
    return bgr[:, :, ::-1]


def open_video_writer(path: Path, fps: float, quality: int) -> imageio.Writer:
    """確認用MP4 writerを開く。"""
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=1,
    )


def generate_dataset(args: argparse.Namespace) -> None:
    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)

    src_dir = args.output / "src"
    test_dir = args.output / "test"
    gt_dir = args.output / "gt"
    src_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    world_width = args.width + 96
    world_height = args.height + 96
    if args.background_mode == "skimage":
        world = make_skimage_world(world_width, world_height, rng, args.skimage_image)
    else:
        world = make_static_world(world_width, world_height, rng)
    objects = [
        MovingObject("ellipse", (40.0, 50.0), (86.0, 58.0), (58.0, 20.0), (0.30, 0.62, 0.78)),
        MovingObject("box", (args.width * 0.55, args.height * 0.54), (92.0, 74.0), (-36.0, 18.0), (0.76, 0.38, 0.24)),
        MovingObject("line", (args.width * 0.25, args.height * 0.72), (150.0, 34.0), (26.0, -12.0), (0.18, 0.78, 0.32)),
    ]

    fixed_pattern = make_low_frequency_noise(rng, args.height, args.width, 3, 9)
    fixed_pattern += rng.normal(0.0, 1.0, size=(args.height, args.width, 3)).astype(np.float32) * 0.22
    fixed_pattern *= args.fixed_pattern_strength
    previous_noise = np.zeros((args.height, args.width, 3), dtype=np.float32)
    video_writers = []
    if not args.no_videos:
        video_writers = [
            (open_video_writer(args.output / "gt_video.mp4", args.fps, args.video_quality), "gt"),
            (open_video_writer(args.output / "src_video.mp4", args.fps, args.video_quality), "src"),
            (open_video_writer(args.output / "test_video.mp4", args.fps, args.video_quality), "test"),
        ]

    try:
        for frame_index in range(args.frames):
            clean = make_clean_frame(world, objects, args.width, args.height, frame_index, args.fps)
            noisy, previous_noise = add_sensor_noise(
                clean,
                rng,
                fixed_pattern,
                previous_noise,
                args.temporal_correlation,
                args.noise_strength,
            )

            name = f"frame_{frame_index:04d}.tiff"
            write_tiff(gt_dir / name, clean, args.bit_depth)
            write_tiff(src_dir / name, noisy, args.bit_depth)
            write_tiff(test_dir / name, noisy, args.bit_depth)

            for writer, kind in video_writers:
                frame = clean if kind == "gt" else noisy
                writer.append_data(video_frame_u8(frame))
    finally:
        for writer, _ in video_writers:
            writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic video-like TIFF16 validation dataset with src/test/gt folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output", type=Path, help="new output dataset directory")
    parser.add_argument("--width", type=int, default=960, help="frame width")
    parser.add_argument("--height", type=int, default=540, help="frame height")
    parser.add_argument("--frames", type=int, default=120, help="number of frames")
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second used for motion")
    parser.add_argument("--bit-depth", type=int, choices=(8, 10, 12, 14, 16), default=16)
    parser.add_argument(
        "--background-mode",
        choices=("synthetic", "skimage"),
        default="synthetic",
        help="background source mode",
    )
    parser.add_argument(
        "--skimage-image",
        choices=skimage_image_choices(),
        default="auto",
        help="scikit-image sample image used when --background-mode skimage",
    )
    parser.add_argument("--noise-strength", type=float, default=0.020, help="normalized noise strength")
    parser.add_argument("--fixed-pattern-strength", type=float, default=0.0025, help="normalized fixed-pattern noise strength")
    parser.add_argument("--temporal-correlation", type=float, default=0.18, help="0..1 noise temporal correlation")
    parser.add_argument("--seed", type=int, default=20260524, help="random seed")
    parser.add_argument("--video-quality", type=int, default=8, help="preview MP4 quality, 0..10")
    parser.add_argument("--no-videos", action="store_true", help="do not write gt/src/test preview MP4 files")
    parser.add_argument("--overwrite", action="store_true", help="replace output directory if it exists")
    return parser


def main() -> int:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if args.frames < 2:
        parser.error("frames must be at least 2")
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.noise_strength < 0:
        parser.error("noise-strength must be >= 0")
    if args.fixed_pattern_strength < 0:
        parser.error("fixed-pattern-strength must be >= 0")
    if not 0.0 <= args.temporal_correlation <= 1.0:
        parser.error("temporal-correlation must be 0.0..1.0")
    if not 0 <= args.video_quality <= 10:
        parser.error("video-quality must be 0..10")

    try:
        generate_dataset(args)
    except RuntimeError as exc:
        parser.error(str(exc))

    print(
        f"wrote {args.output} "
        f"(src/test/gt, frames={args.frames}, size={args.width}x{args.height}, "
        f"bit_depth={args.bit_depth}, background_mode={args.background_mode}, "
        f"videos={not args.no_videos})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
