"""复现论文第IV节的主要实验流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from sbl import (
    SBLConfig,
    SparseBayesianLearning,
    beta_metric,
    generate_line_spectrum,
    normalized_mse,
)


@dataclass
class ExperimentResult:
    """单次实验的统计结果。"""

    mse: float
    beta: float
    estimated_order: int


def run_single_trial(config: SBLConfig, snr_db: float, rng: np.random.Generator) -> ExperimentResult:
    """运行一次仿真实验并返回结果。"""

    from sbl.simulation import LineSpectrumConfig

    spectrum_config = LineSpectrumConfig(snr_db=snr_db)
    data = generate_line_spectrum(spectrum_config, rng=rng)

    model = SparseBayesianLearning(num_samples=spectrum_config.num_samples, config=config)
    model.fit(data.noisy_observation)

    reconstructed = model.reconstruct_signal()
    mse = normalized_mse(data.signal, reconstructed)
    beta = beta_metric(data.thetas, model.thetas_)
    estimated_order = 0 if model.thetas_ is None else len(model.thetas_)
    return ExperimentResult(mse=mse, beta=beta, estimated_order=estimated_order)


def run_experiment(
    snr_db_list: List[float],
    trials_per_snr: int = 10,
    seed: int | None = 0,
) -> None:
    """对一系列SNR进行重复实验并打印统计指标。"""

    rng = np.random.default_rng(seed)
    config = SBLConfig()

    for snr_db in snr_db_list:
        mses = []
        betas = []
        orders = []
        for _ in range(trials_per_snr):
            result = run_single_trial(config, snr_db, rng)
            mses.append(result.mse)
            betas.append(result.beta)
            orders.append(result.estimated_order)
        print(f"SNR={snr_db:.1f} dB: MSE={np.mean(mses):.4e}, beta={np.mean(betas):.4e}, order={np.mean(orders):.2f}")


if __name__ == "__main__":
    snr_values = [0, 5, 10, 15, 20]
    run_experiment(snr_values, trials_per_snr=5)
