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


@dataclass
class SyntheticOFDMDataset:
    """由仿真生成的OFDM频域响应及对应真值。"""

    data: np.ndarray
    """形状为``(num, 1, symbol, antenna, freq)``的复值频响。"""

    true_toas: List[np.ndarray]
    """每个样本的真实多径到达时间(秒)。"""

    subcarrier_spacing_hz: float
    """生成数据时使用的子载波间隔。"""


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


def generate_synthetic_ofdm_dataset(
    num_samples: int = 4,
    num_symbols: int = 1,
    num_antennas: int = 1,
    num_subcarriers: int = 480,
    subcarrier_spacing_hz: float = 120_000.0,
    max_paths: int = 3,
    delay_range_s: tuple[float, float] = (50e-9, 5e-6),
    snr_db: float = 30.0,
    seed: int | None = None,
) -> SyntheticOFDMDataset:
    """生成满足接口要求的稀疏OFDM频域响应数据。

    该函数模拟若干条多径信道, 每条路径的到达时间在``delay_range_s``内均匀
    采样, 幅度采用幅度衰减+随机相位模型, 并向观测频响加入复高斯噪声。
    返回的数据可直接用于 :func:`estimate_toas_for_dataset` 。
    """

    rng = np.random.default_rng(seed)
    noise_variance = 0.0
    if np.isfinite(snr_db):
        noise_variance = 10.0 ** (-snr_db / 10.0)

    true_toas: List[np.ndarray] = []
    data = np.zeros(
        (num_samples, 1, num_symbols, num_antennas, num_subcarriers),
        dtype=np.complex128,
    )

    frequency_indices = np.arange(num_subcarriers, dtype=float)
    frequencies = frequency_indices * subcarrier_spacing_hz

    for sample_idx in range(num_samples):
        num_paths = rng.integers(1, max_paths + 1)
        toas = rng.uniform(delay_range_s[0], delay_range_s[1], size=num_paths)
        toas.sort()

        amplitudes = rng.rayleigh(scale=1.0, size=num_paths)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=num_paths)
        complex_gains = amplitudes * np.exp(1j * phases)

        response = np.zeros(num_subcarriers, dtype=np.complex128)
        for gain, toa in zip(complex_gains, toas):
            response += gain * np.exp(-1j * 2.0 * np.pi * frequencies * toa)

        if noise_variance > 0.0:
            noise = rng.normal(scale=np.sqrt(noise_variance / 2.0), size=response.shape)
            noise = noise + 1j * rng.normal(
                scale=np.sqrt(noise_variance / 2.0), size=response.shape
            )
            response += noise

        # 仅填充第一个符号与天线, 其余位置留为零便于后续自定义
        data[sample_idx, 0, 0, 0, :] = response
        true_toas.append(toas.astype(float))

    return SyntheticOFDMDataset(
        data=data,
        true_toas=true_toas,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
    )
