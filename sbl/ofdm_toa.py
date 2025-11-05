"""OFDM场景下基于频域导频的SBL多径TOA提取流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .sbl_algorithm import SBLConfig, SparseBayesianLearning


@dataclass
class SnapshotTOAResult:
    """单个快照(单条num数据)的多径估计结果。"""

    toas: np.ndarray
    """以秒为单位的多径到达时间。"""

    thetas: np.ndarray
    """SBL返回的归一化参数, 与``toas``满足``theta = toa * subcarrier_spacing``。"""

    amplitudes: np.ndarray
    """后验均值幅度, 用于表示各条路径的能量。"""

    gammas: np.ndarray
    """稀疏先验的尺度参数, 也可用于判定有效路径。"""


def _prepare_observation(frequency_response: np.ndarray) -> np.ndarray:
    """将频域导频最小二乘估计转换为SBL模型可处理的形式。"""

    y = np.asarray(frequency_response, dtype=complex)
    if y.ndim != 1:
        y = y.reshape(-1)
    # 频域模型为 exp(-j2π f τ), 取共轭即可匹配 exp(j2π θ n) 的实现
    return np.conj(y)


def _extract_snapshot_toa(
    frequency_response: np.ndarray,
    subcarrier_spacing_hz: float,
    config: SBLConfig | None,
) -> SnapshotTOAResult:
    """对单条频响数据执行SBL, 获得TOA估计。"""

    observation = _prepare_observation(frequency_response)
    model = SparseBayesianLearning(num_samples=observation.size, config=config)
    model.fit(observation)

    if model.thetas_ is None or model.gammas_ is None:
        return SnapshotTOAResult(
            toas=np.array([], dtype=float),
            thetas=np.array([], dtype=float),
            amplitudes=np.array([], dtype=float),
            gammas=np.array([], dtype=float),
        )

    thetas = np.asarray(model.thetas_, dtype=float)
    gammas = np.asarray(model.gammas_, dtype=float)
    if thetas.size == 0:
        return SnapshotTOAResult(
            toas=np.array([], dtype=float),
            thetas=thetas,
            amplitudes=np.array([], dtype=float),
            gammas=gammas,
        )

    toas = thetas / subcarrier_spacing_hz
    order = np.argsort(toas)
    toas = toas[order]
    thetas = thetas[order]
    gammas = gammas[order]

    if model.alphas_ is not None and model.alphas_.size == thetas.size:
        amplitudes = np.abs(np.asarray(model.alphas_)[order])
    else:
        amplitudes = np.zeros_like(toas)

    return SnapshotTOAResult(
        toas=toas,
        thetas=thetas,
        amplitudes=amplitudes,
        gammas=gammas,
    )


def estimate_toas_for_dataset(
    data: np.ndarray,
    subcarrier_spacing_hz: float,
    symbol_index: int = 0,
    antenna_index: int = 0,
    config: SBLConfig | None = None,
) -> List[SnapshotTOAResult]:
    """对数据集中所有``num``样本估计多径TOA。"""

    if data.ndim != 5:
        raise ValueError("输入数据维度应为(num, 1, symbol, att, freq)")

    num_samples = data.shape[0]
    results: List[SnapshotTOAResult] = []

    for idx in range(num_samples):
        frequency_response = data[idx, 0, symbol_index, antenna_index, :]
        result = _extract_snapshot_toa(frequency_response, subcarrier_spacing_hz, config)
        results.append(result)

    return results


def stack_toa_arrays(results: Sequence[SnapshotTOAResult]) -> List[np.ndarray]:
    """将TOA结果转换为便于绘图或后续处理的列表形式。"""

    stacked: List[np.ndarray] = []
    for res in results:
        stacked.append(np.asarray(res.toas, dtype=float))
    return stacked


def filter_paths_by_gamma(
    results: Sequence[SnapshotTOAResult],
    gamma_threshold: float,
) -> List[SnapshotTOAResult]:
    """根据``gamma``阈值筛除能量较弱的路径。"""

    filtered: List[SnapshotTOAResult] = []
    for res in results:
        mask = res.gammas >= gamma_threshold
        filtered.append(
            SnapshotTOAResult(
                toas=res.toas[mask],
                thetas=res.thetas[mask],
                amplitudes=res.amplitudes[mask],
                gammas=res.gammas[mask],
            )
        )
    return filtered
