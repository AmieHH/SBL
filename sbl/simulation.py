"""用于生成与论文实验一致的仿真数据。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atoms import fourier_atom


@dataclass
class LineSpectrumConfig:
    """线谱信号的参数配置。"""

    num_samples: int = 100
    num_components: int = 10
    snr_db: float = 15.0


@dataclass
class LineSpectrumData:
    """线谱信号及其标签数据。"""

    signal: np.ndarray
    noisy_observation: np.ndarray
    thetas: np.ndarray
    alphas: np.ndarray
    noise_variance: float


def _generate_theta_pairs(config: LineSpectrumConfig, rng: np.random.Generator) -> np.ndarray:
    """生成满足论文实验设定的成对频率。"""

    N = config.num_samples
    K = config.num_components
    assert K % 2 == 0, "按照论文设定,K需要是偶数"

    pairs = []
    # 每对频率间距在[0.7/N, 1/N], 其余频率间距至少1.5/N
    min_pair_gap = 0.7 / N
    max_pair_gap = 1.0 / N
    min_inter_pair_gap = 1.5 / N

    centers = []
    while len(centers) < K // 2:
        candidate = rng.uniform(0.0, 1.0)
        if all(
            min(
                abs(candidate - c),
                1.0 - abs(candidate - c),
            ) >= min_inter_pair_gap
            for c in centers
        ):
            centers.append(candidate)
    centers = np.array(centers)

    for center in centers:
        gap = rng.uniform(min_pair_gap, max_pair_gap)
        theta1 = (center - gap / 2.0) % 1.0
        theta2 = (center + gap / 2.0) % 1.0
        pairs.extend([theta1, theta2])

    return np.sort(np.array(pairs))


def generate_line_spectrum(
    config: LineSpectrumConfig, rng: np.random.Generator | None = None
) -> LineSpectrumData:
    """生成论文实验中使用的线谱信号及观测。"""

    if rng is None:
        rng = np.random.default_rng()

    N = config.num_samples
    K = config.num_components

    thetas = _generate_theta_pairs(config, rng)
    amplitudes = rng.normal(loc=1.0, scale=np.sqrt(0.1), size=K)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=K)
    alphas = amplitudes * np.exp(1j * phases)

    signal = np.zeros(N, dtype=complex)
    for theta, alpha in zip(thetas, alphas):
        signal += alpha * fourier_atom(theta, N)

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    noise_std = np.sqrt(noise_variance / 2.0)
    noise = noise_std * (rng.normal(size=N) + 1j * rng.normal(size=N))

    noisy_observation = signal + noise
    return LineSpectrumData(
        signal=signal,
        noisy_observation=noisy_observation,
        thetas=thetas,
        alphas=alphas,
        noise_variance=noise_variance,
    )
