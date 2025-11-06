"""将SBL算法应用于OFDM导频频域数据, 提取多径TOA并绘图。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from sbl import (
    SBLConfig,
    SyntheticOFDMDataset,
    estimate_toas_for_dataset,
    filter_paths_by_gamma,
    generate_synthetic_ofdm_dataset,
)

_TIME_UNITS: Dict[str, float] = {
    "s": 1.0,
    "ms": 1e3,
    "us": 1e6,
    "ns": 1e9,
}


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="使用稀疏贝叶斯学习在OFDM频响数据上估计多径TOA"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        help=".npy格式的频域最小二乘估计数据路径",
    )
    parser.add_argument(
        "--subcarrier-spacing",
        type=float,
        required=True,
        help="子载波间隔(Hz), 例如120000表示120kHz",
    )
    parser.add_argument(
        "--symbol-index",
        type=int,
        default=0,
        help="用于估计的符号序号, 默认取第0个",
    )
    parser.add_argument(
        "--antenna-index",
        type=int,
        default=0,
        help="用于估计的天线序号, 默认取第0个",
    )
    parser.add_argument(
        "--gamma-threshold",
        type=float,
        default=0.0,
        help="可选的γ阈值, 过滤掉能量较小的路径",
    )
    parser.add_argument(
        "--time-unit",
        choices=tuple(_TIME_UNITS.keys()),
        default="us",
        help="绘图使用的时间单位, 默认微秒(us)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0,
        help="SBL中的ε超参数, 控制稀疏程度",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="SBL主循环的最大迭代次数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="可选随机种子, 用于复现仿真数据",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="启用内置仿真, 无需加载真实数据",
    )
    parser.add_argument(
        "--simulate-num-samples",
        type=int,
        default=4,
        help="仿真样本数量(num维度)",
    )
    parser.add_argument(
        "--simulate-max-paths",
        type=int,
        default=3,
        help="单个样本的最大多径条数",
    )
    parser.add_argument(
        "--simulate-snr-db",
        type=float,
        default=30.0,
        help="仿真观测的信噪比, 单位dB",
    )
    parser.add_argument(
        "--simulate-save-path",
        type=Path,
        help="若指定则把仿真数据保存为.npy文件",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="若指定则将散点图保存到该路径",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否在屏幕上展示图像",
    )
    return parser


def _plot_toa(results, time_unit: str, save_path: Path | None, show: bool) -> None:
    """根据TOA结果绘制散点图。"""

    scale = _TIME_UNITS[time_unit]
    xs = []
    ys = []
    colors = []

    for idx, res in enumerate(results):
        if res.toas.size == 0:
            continue
        print(res.toas)
        xs.extend([idx] * res.toas.size)
        ys.extend(res.toas * scale)
        colors.extend(res.amplitudes)

    plt.figure(figsize=(10, 4))
    if xs:
        scatter = plt.scatter(xs, ys, c=colors, cmap="viridis", s=25, edgecolor="none")
        cbar = plt.colorbar(scatter)
        cbar.set_label("|α| (后验幅度)")
    else:
        plt.scatter([], [])
    plt.xlabel("num样本索引")
    plt.ylabel(f"到达时间 / {time_unit}")
    plt.title("SBL提取的多径到达时间")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_path='./ofdm_toa_estimation.png'
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    synthetic: SyntheticOFDMDataset | None = None
    if args.simulate:
        synthetic = generate_synthetic_ofdm_dataset(
            num_samples=args.simulate_num_samples,
            num_symbols=1,
            num_antennas=1,
            num_subcarriers=480,
            subcarrier_spacing_hz=args.subcarrier_spacing,
            max_paths=args.simulate_max_paths,
            snr_db=args.simulate_snr_db,
            seed=args.seed,
        )
        data = synthetic.data
        if args.simulate_save_path is not None:
            np.save(args.simulate_save_path, data)
    else:
        if args.data_path is None:
            raise SystemExit("未提供数据路径, 如需仿真请添加 --simulate 参数")
        data = np.load(args.data_path)

    if data.ndim != 5:
        raise ValueError("数据维度应为(num, 1, symbol, att, freq)")

    config = SBLConfig(epsilon=args.epsilon, max_iterations=args.max_iterations)
    results = estimate_toas_for_dataset(
        data,
        subcarrier_spacing_hz=args.subcarrier_spacing,
        symbol_index=args.symbol_index,
        antenna_index=args.antenna_index,
        config=config,
    )

    if args.gamma_threshold > 0.0:
        results = filter_paths_by_gamma(results, args.gamma_threshold)
    
    _plot_toa(results, args.time_unit, args.save_path, args.show)

    if synthetic is not None:
        print("仿真真值TOA(秒):")
        for idx, toas in enumerate(synthetic.true_toas):
            formatted = ", ".join(f"{toa:.3e}" for toa in toas)
            print(f"  样本{idx}: {formatted}")


if __name__ == "__main__":
    main()
