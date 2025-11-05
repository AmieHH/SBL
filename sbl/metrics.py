"""评估指标函数。"""

from __future__ import annotations

import numpy as np


def normalized_mse(true_signal: np.ndarray, estimated_signal: np.ndarray) -> float:
    """计算归一化均方误差(MSE)。"""

    numerator = np.linalg.norm(true_signal - estimated_signal) ** 2
    denominator = np.linalg.norm(true_signal) ** 2 + 1e-12
    return float(numerator / denominator)


def beta_metric(true_thetas: np.ndarray, estimated_thetas: np.ndarray, period: float = 1.0) -> float:
    """计算论文中的频率估计误差``\beta``指标。"""

    if len(estimated_thetas) == 0:
        return float(np.inf)

    diffs = []
    for theta in true_thetas:
        wrapped_diff = np.min(np.mod(np.abs(theta - estimated_thetas), period))
        wrapped_diff = min(wrapped_diff, period - wrapped_diff)
        diffs.append(wrapped_diff**2)
    return float(np.mean(diffs))
