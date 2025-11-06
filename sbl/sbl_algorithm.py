"""稀疏贝叶斯学习(SBL)主算法实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .atoms import (
    fourier_atom,
    fourier_atom_first_derivative,
    fourier_atom_second_derivative,
)


@dataclass
class SBLConfig:
    """SBL算法的超参数设置。"""

    # epsilon: float = 0.5
    # a: float = 1e-6
    # b: float = 1e-6
    epsilon: float = 0
    a: float = 0
    b: float = 0
    rho: float = 1.0  # 复数情形
    grid_factor: int = 3
    max_iterations: int = 50
    newton_max_iterations: int = 1 #15
    newton_tolerance: float = 1e-8


class SparseBayesianLearning:
    """实现论文中的SBL字典参数估计算法。"""

    def __init__(self, num_samples: int, config: SBLConfig | None = None):
        self.num_samples = num_samples
        self.config = config or SBLConfig()
        self.thetas_: np.ndarray | None = None
        self.gammas_: np.ndarray | None = None
        self.lambda_: float | None = None
        self.alphas_: np.ndarray | None = None

    # ========================= 核心矩阵运算 ========================= #
    def _compute_dictionary(self, thetas: np.ndarray) -> np.ndarray:
        """根据当前频率向量构建字典矩阵。"""

        if len(thetas) == 0:
            return np.zeros((self.num_samples, 0), dtype=complex)
        columns = [fourier_atom(theta, self.num_samples) for theta in thetas]
        return np.column_stack(columns)

    def _compute_B_inverse(self, thetas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
        """计算矩阵``B = I + Psi Gamma Psi^H``的逆。"""

        identity = np.eye(self.num_samples, dtype=complex)
        if len(thetas) == 0:
            return identity
        psi = self._compute_dictionary(thetas)
        gamma_matrix = np.diag(gammas)
        B = identity + psi @ gamma_matrix @ psi.conj().T
        return np.linalg.inv(B)

    def _statistics_for_theta(
        self,
        theta: float,
        B_inv: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, complex, float, float, float, float]:
        """计算公式中涉及的统计量及导数。"""

        psi = fourier_atom(theta, self.num_samples)
        dpsi = fourier_atom_first_derivative(theta, self.num_samples)
        d2psi = fourier_atom_second_derivative(theta, self.num_samples)

        B_inv_psi = B_inv @ psi
        B_inv_dpsi = B_inv @ dpsi
        B_inv_d2psi = B_inv @ d2psi
        B_inv_y = B_inv @ y

        s = np.real(np.vdot(psi, B_inv_psi))
        q = np.vdot(psi, B_inv_y)

        ds = 2.0 * np.real(np.vdot(psi, B_inv_dpsi))
        dq_abs = 2.0 * np.real(np.conj(q) * np.vdot(y, B_inv_dpsi))

        d2s = 2.0 * np.real(np.vdot(psi, B_inv_d2psi) + np.vdot(dpsi, B_inv_dpsi))

        temp = np.vdot(y, B_inv_dpsi)
        temp2 = np.vdot(y, B_inv_d2psi)
        temp3 = np.vdot(y, B_inv_psi)
        d2q_abs = 2.0 * np.real(temp * np.conj(temp) + temp2 * np.conj(temp3))

        return s, q, ds, dq_abs, d2s, d2q_abs

    # ========================= 参数更新 ========================= #
    def _update_gamma(
        self,
        lambda_hat: float,
        s_hat: float,
        q_hat: complex,
    ) -> float:
        """根据公式(12)更新单个``gamma``。"""

        eps = self.config.epsilon
        rho = self.config.rho
        abs_q_sq = np.abs(q_hat) ** 2

        threshold_term = (
            2.0
            + rho
            - 2.0 * eps
            + 2.0 * np.sqrt(max((1.0 - eps) * (1.0 + rho - eps), 0.0))
        ) * s_hat
        lhs = rho * lambda_hat * abs_q_sq
        if lhs <= threshold_term or s_hat <= 0:
            return 0.0

        delta = (
            (2.0 * eps - 2.0 - rho) * s_hat + rho * lambda_hat * abs_q_sq
        ) ** 2 - 4.0 * (eps - 1.0) * (eps - 1.0 - rho) * (s_hat**2)
        delta = max(delta, 0.0)

        denominator = 2.0 * (eps - 1.0 - rho) * (s_hat**2)
        if np.isclose(denominator, 0.0):
            return 0.0

        numerator = (
            -(2.0 * eps - 2.0 - rho) * s_hat
            - rho * lambda_hat * abs_q_sq
            - np.sqrt(delta)
        )
        gamma = numerator / denominator
        if not np.isfinite(gamma) or gamma <= 0:
            return 0.0
        return float(np.real(gamma))

    def _gradient_hessian(
        self,
        gamma_hat: float,
        lambda_hat: float,
        s_hat: float,
        q_hat: complex,
        ds_hat: float,
        dq_abs_hat: float,
        d2s_hat: float,
        d2q_abs_hat: float,
    ) -> Tuple[float, float]:
        """计算目标函数对``theta``的一阶和二阶导数。"""

        rho = self.config.rho
        abs_q_sq = np.abs(q_hat) ** 2
        denom = 1.0 + gamma_hat * s_hat

        grad = (
            (rho * lambda_hat * gamma_hat / denom) * dq_abs_hat
            - (
                (rho * gamma_hat / denom)
                + (rho * lambda_hat * (gamma_hat**2) * abs_q_sq) / (denom**2)
            )
            * ds_hat
        )

        hessian = (
            (lambda_hat * d2q_abs_hat - d2s_hat) * (rho * gamma_hat / denom)
            + (ds_hat**2) * (2.0 * rho * lambda_hat * (gamma_hat**3) * abs_q_sq) / (denom**3)
            + (
                (ds_hat**2)
                - 2.0 * lambda_hat * ds_hat * dq_abs_hat
                - lambda_hat * abs_q_sq * d2s_hat
            )
            * (rho * (gamma_hat**2) / (denom**2))
        )

        return float(np.real(grad)), float(np.real(hessian))

    def _newton_refine(
        self,
        theta_init: float,
        lambda_hat: float,
        B_inv: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """对单个频率执行牛顿迭代并返回更新后的``(theta, gamma)``。"""

        theta = float(theta_init % 1.0)
        gamma = 0.0
        for _ in range(self.config.newton_max_iterations):
            stats = self._statistics_for_theta(theta, B_inv, y)
            s_hat, q_hat, ds_hat, dq_abs_hat, d2s_hat, d2q_abs_hat = stats
            gamma = self._update_gamma(lambda_hat, s_hat, q_hat)
            if gamma <= 0.0:
                return theta, 0.0
            grad, hess = self._gradient_hessian(
                gamma,
                lambda_hat,
                s_hat,
                q_hat,
                ds_hat,
                dq_abs_hat,
                d2s_hat,
                d2q_abs_hat,
            )
            if abs(hess) < 1e-12:
                break
            step = grad / hess
            theta_new = (theta - step) % 1.0
            diff = min(abs(theta_new - theta), 1.0 - abs(theta_new - theta))
            theta = theta_new
            if diff < self.config.newton_tolerance:
                break
        # 重新评估gamma以保证返回值与最终theta匹配
        stats = self._statistics_for_theta(theta, B_inv, y)
        s_hat, q_hat, *_ = stats
        gamma = self._update_gamma(lambda_hat, s_hat, q_hat)
        return theta % 1.0, gamma

    # ========================= 外部接口 ========================= #
    def fit(self, y: np.ndarray) -> "SparseBayesianLearning":
        """根据观测向量估计频率、系数以及噪声精度。"""

        y = np.asarray(y, dtype=complex)
        assert y.shape[0] == self.num_samples, "观测长度与模型配置不一致"

        lambda_hat = 100.0
        thetas: List[float] = []
        gammas: List[float] = []

        prev_thetas = np.array([])

        for iteration in range(self.config.max_iterations):
            # 周期性执行网格搜索以引入新原子
            if iteration % 5 == 0:
                B_inv_base = self._compute_B_inverse(np.array(thetas), np.array(gammas))
                for idx in range(self.config.grid_factor * self.num_samples):
                    theta_init = idx / (self.config.grid_factor * self.num_samples)
                    theta_cand, gamma_cand = self._newton_refine(
                        theta_init, lambda_hat, B_inv_base, y
                    )
                    if gamma_cand <= 0.0:
                        continue
                    if thetas:
                        min_distance = min(
                            min(
                                abs(theta_cand - t),
                                1.0 - abs(theta_cand - t),
                            )
                            for t in thetas
                        )
                        if min_distance < 1e-3:
                            continue
                    thetas.append(theta_cand)
                    gammas.append(gamma_cand)

            # 更新已有原子
            index = 0
            while index < len(thetas):
                current_thetas = np.array(thetas)
                current_gammas = np.array(gammas)
                theta_i = current_thetas[index]
                gamma_i = current_gammas[index]

                mask = np.ones(len(thetas), dtype=bool)
                mask[index] = False
                reduced_thetas = current_thetas[mask]
                reduced_gammas = current_gammas[mask]
                B_inv_minus = self._compute_B_inverse(reduced_thetas, reduced_gammas)
                theta_new, gamma_new = self._newton_refine(
                    theta_i, lambda_hat, B_inv_minus, y
                )
                if gamma_new <= 0.0:
                    thetas.pop(index)
                    gammas.pop(index)
                    continue
                thetas[index] = theta_new
                gammas[index] = gamma_new
                index += 1

            if len(thetas) == 0:
                B_inv_total = np.eye(self.num_samples, dtype=complex)
            else:
                B_inv_total = self._compute_B_inverse(np.array(thetas), np.array(gammas))

            numerator = self.config.rho * self.num_samples + self.config.a - 1.0
            denominator = self.config.rho * np.real(np.vdot(y, B_inv_total @ y)) - self.config.b
            if denominator <= 0:
                break
            lambda_hat = numerator / denominator

            current_thetas = np.array(thetas)
            if prev_thetas.size == current_thetas.size and current_thetas.size > 0:
                diffs = [
                    min(abs(a - b), 1.0 - abs(a - b))
                    for a, b in zip(np.sort(prev_thetas), np.sort(current_thetas))
                ]
                if diffs and max(diffs) < (1e-6 / self.num_samples):
                    break
            prev_thetas = current_thetas.copy()

        self.thetas_ = np.array(thetas)
        self.gammas_ = np.array(gammas)
        self.lambda_ = float(lambda_hat)

        if len(thetas) == 0:
            self.alphas_ = np.array([])
            return self

        psi = self._compute_dictionary(self.thetas_)
        gamma_matrix = np.diag(self.gammas_)
        sigma = np.linalg.inv(psi.conj().T @ psi + np.linalg.inv(gamma_matrix))
        mu = sigma @ psi.conj().T @ y
        self.alphas_ = mu
        return self

    def reconstruct_signal(self) -> np.ndarray:
        """利用估计的参数重构信号。"""

        if self.thetas_ is None or self.alphas_ is None:
            raise RuntimeError("请先调用fit方法进行估计")
        if len(self.thetas_) == 0:
            return np.zeros(self.num_samples, dtype=complex)
        psi = self._compute_dictionary(self.thetas_)
        return psi @ self.alphas_
