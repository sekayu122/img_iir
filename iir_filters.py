"""Reusable frame filters for image sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod

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

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be 0.0..1.0")
        self.alpha = float(alpha)
        self._previous_output: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_output = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self._previous_output is None:
            output = frame.astype(np.float32)
        else:
            output = (
                frame.astype(np.float32) * self.alpha
                + self._previous_output * (1.0 - self.alpha)
            )
        self._previous_output = output
        return output


def create_filter(name: str, alpha: float) -> FrameFilter:
    """名前からフィルタインスタンスを作る。"""
    if name == "alpha":
        return AlphaBlendIirFilter(alpha)
    raise ValueError(f"unknown filter: {name}")
