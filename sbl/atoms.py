"""频率原子函数及其导数的实现。"""

from __future__ import annotations

import numpy as np


def _index_vector(length: int) -> np.ndarray:
    """生成0到``length-1``的整数索引向量。"""

    return np.arange(length, dtype=float)


def fourier_atom(theta: float, length: int) -> np.ndarray:
    """计算傅里叶原子向量。

    参数
    ------
    theta:
        归一化频率, 取值范围为``[0, 1)``。
    length:
        原子向量长度, 即观测长度``N``。
    返回
    ------
    np.ndarray
        形状为``(length,)``的复数向量``psi(theta)``。
    """

    n = _index_vector(length)
    return np.exp(1j * 2.0 * np.pi * theta * n)


def fourier_atom_first_derivative(theta: float, length: int) -> np.ndarray:
    """计算傅里叶原子关于频率的一阶导数。"""

    n = _index_vector(length)
    base = fourier_atom(theta, length)
    return 1j * 2.0 * np.pi * n * base


def fourier_atom_second_derivative(theta: float, length: int) -> np.ndarray:
    """计算傅里叶原子关于频率的二阶导数。"""

    n = _index_vector(length)
    base = fourier_atom(theta, length)
    return -(2.0 * np.pi) ** 2 * (n**2) * base
