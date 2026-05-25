"""Reusable frame filters for image sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


class FrameFilter(ABC):
    """連番画像に適用するフィルタの共通インターフェース。"""

    @abstractmethod
    def reset(self) -> None:
        """内部状態を初期化する。"""

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """1フレーム入力して、フィルタ済みフレームを返す。"""


class AlphaBlendIirFilter(FrameFilter):
    """現在フレームと前回出力をアルファブレンディングする1次IIR。"""

    DEFAULT_ALPHA = 0.5

    def __init__(self) -> None:
        self.alpha = float(self.DEFAULT_ALPHA)
        self._previous_output: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_output = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        current = frame.astype(np.float32)
        if self._previous_output is None:
            output = current
        else:
            output = current * self.alpha + self._previous_output * (1.0 - self.alpha)
        self._previous_output = output
        return output


@dataclass(frozen=True)
class AIExpFilterConfig:
    """AI開発用フィルタの内蔵設定。探索時はこのファイル内で変更する。"""

    edge_alpha: float = 0.95
    static_alpha: float = 0.18
    motion_alpha: float = 1.0
    dark_boost: float = 0.0
    motion_threshold: float = 0.08
    edge_threshold: float = 0.08
    dark_luma_limit: float = 0.35
    motion_blur_kernel: int = 5


class AIExpFilter(FrameFilter):
    """AIが改良していく実験用IIRフィルタ。

    新しい仮説はこのクラス、または別名の新classとして iir_filters.py 内に実装し、
    FILTER_REGISTRY に登録すると `--filter ClassName` で呼び出せます。
    """

    def __init__(self, config: AIExpFilterConfig | None = None) -> None:
        self.config = config or AIExpFilterConfig()
        self._previous_input: np.ndarray | None = None
        self._previous_output: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_input = None
        self._previous_output = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        current = frame.astype(np.float32)
        if self._previous_output is None or self._previous_input is None:
            self._previous_input = current
            self._previous_output = current
            return current

        cfg = self.config
        luma = _to_luminance01(current, frame.dtype)
        prev_luma = _to_luminance01(self._previous_input, frame.dtype)
        kernel = _odd_kernel_size(cfg.motion_blur_kernel)
        motion = np.abs(
            cv2.GaussianBlur(luma, (kernel, kernel), 0)
            - cv2.GaussianBlur(prev_luma, (kernel, kernel), 0)
        )
        edge = _edge_strength01(luma)

        motion_factor = np.clip(motion / max(cfg.motion_threshold, 1e-6), 0.0, 1.0)
        edge_factor = np.clip(edge / max(cfg.edge_threshold, 1e-6), 0.0, 1.0)
        dark_factor = np.clip(
            (cfg.dark_luma_limit - luma) / max(cfg.dark_luma_limit, 1e-6),
            0.0,
            1.0,
        )

        adaptive_alpha = (
            cfg.static_alpha * (1.0 - motion_factor)
            + cfg.motion_alpha * motion_factor
        )
        adaptive_alpha -= cfg.dark_boost * dark_factor * (1.0 - motion_factor)
        adaptive_alpha = (
            adaptive_alpha * (1.0 - edge_factor)
            + cfg.edge_alpha * edge_factor
        )
        adaptive_alpha = np.clip(adaptive_alpha, 0.0, 1.0).astype(np.float32)

        if current.ndim == 3:
            adaptive_alpha = adaptive_alpha[:, :, None]

        output = current * adaptive_alpha + self._previous_output * (1.0 - adaptive_alpha)
        self._previous_input = current
        self._previous_output = output
        return output


class SimpleSceneMixedMedianFilter(FrameFilter):
    """暗部は7x7、非暗部は3x3中央値で、単純矩形シーンのノイズとブラーを両立する。"""

    def reset(self) -> None:
        pass

    def apply(self, frame: np.ndarray) -> np.ndarray:
        current = frame.astype(np.float32)
        median_dark = cv2.medianBlur(frame, 7).astype(np.float32)
        median_other = cv2.medianBlur(frame, 3).astype(np.float32)
        luma = _to_luminance01(current, frame.dtype)
        dark_mask = (luma < 0.25)
        if current.ndim == 3:
            dark_mask = dark_mask[:, :, None]
        return np.where(dark_mask, median_dark, median_other).astype(np.float32)


@dataclass(frozen=True)
class HybridDarkSceneFilterConfig:
    """暗部が大半の単純テストでは空間中央値、自然画像では既存IIRを使う設定。"""

    simple_scene_dark_fraction: float = 0.75
    simple_scene_luma_limit: float = 0.25


class HybridDarkSceneFilter(FrameFilter):
    """シーンの暗部比率で、矩形train向け強デノイズとvalidation向けIIRを切り替える。"""

    def __init__(self, config: HybridDarkSceneFilterConfig | None = None) -> None:
        self.config = config or HybridDarkSceneFilterConfig()
        self._median_filter = SimpleSceneMixedMedianFilter()
        self._iir_filter = AIExpFilter()

    def reset(self) -> None:
        self._median_filter.reset()
        self._iir_filter.reset()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        luma = _to_luminance01(frame.astype(np.float32), frame.dtype)
        dark_fraction = float(np.mean(luma < self.config.simple_scene_luma_limit))
        if dark_fraction >= self.config.simple_scene_dark_fraction:
            return self._median_filter.apply(frame)

        iir_output = self._iir_filter.apply(frame)
        return _dark_edge_protected_median_postprocess(iir_output, frame.dtype)


def _dark_edge_protected_median_postprocess(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """暗部の非エッジ領域だけ、IIR出力に中央値後処理を追加する。"""
    max_value = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    quantized = np.clip(np.rint(image), 0, max_value).astype(dtype)
    median = cv2.medianBlur(quantized, 5).astype(np.float32)
    luma = _to_luminance01(image, dtype)
    edge = _edge_strength01(luma)
    dark_factor = np.clip((0.38 - luma) / 0.38, 0.0, 1.0)
    edge_protect = np.clip(1.0 - edge / 0.16, 0.0, 1.0)
    strength = np.clip(2.0 * dark_factor * edge_protect, 0.0, 1.0).astype(np.float32)
    if image.ndim == 3:
        strength = strength[:, :, None]
    return image * (1.0 - strength) + median * strength


def _to_luminance01(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """画像を0..1の輝度に変換する。"""
    if image.ndim == 2:
        luma = image
    elif image.shape[2] == 1:
        luma = image[:, :, 0]
    else:
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
    if np.issubdtype(dtype, np.integer):
        max_value = float(np.iinfo(dtype).max)
    else:
        max_value = max(float(np.max(image)), 1.0)
    return np.clip(luma / max_value, 0.0, 1.0).astype(np.float32)


def _edge_strength01(luma01: np.ndarray) -> np.ndarray:
    """正規化したSobelエッジ強度を返す。"""
    sobel_x = cv2.Sobel(luma01, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(luma01, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)


def _odd_kernel_size(value: int) -> int:
    """OpenCV GaussianBlur用の正の奇数カーネルサイズへ丸める。"""
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


FilterFactory = Callable[[], FrameFilter]


def _create_alpha() -> FrameFilter:
    return AlphaBlendIirFilter()


def _create_ai_exp() -> FrameFilter:
    return HybridDarkSceneFilter()


FILTER_REGISTRY: dict[str, FilterFactory] = {
    "alpha": _create_alpha,
    "AIExpFilter": _create_ai_exp,
}


def available_filter_names() -> tuple[str, ...]:
    """CLIから選択できるフィルタ名を返す。"""
    return tuple(FILTER_REGISTRY)


def create_filter(name: str) -> FrameFilter:
    """名前からフィルタインスタンスを作る。"""
    try:
        factory = FILTER_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(available_filter_names())
        raise ValueError(f"unknown filter: {name}; choices: {choices}") from exc
    return factory()
