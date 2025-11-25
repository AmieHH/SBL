# SBL算法

单页PPT可用以下要点，突出方法核心。

- **目标**：利用OFDM导频的频域响应稀疏重构多径的到达时间（TOA）。
- **观测模型**：\(H[k]=\sum_i a_i e^{-j2\pi f_k \tau_i}+w[k]\)，其中 \(f_k=k\Delta f\)，噪声为复高斯。
- **稀疏先验**：系数 \(a_i\) 服从零均值复高斯，方差 \(\gamma_i\) 自适应；延迟以角度 \(\theta_i=\Delta f\,\tau_i\) 表征。
- **指数字典**：\(\Psi(:,i)=e^{j2\pi\theta_i k}\) 覆盖候选延迟区间，允许对活跃原子做牛顿式角度微调以突破固定网格。
- **贝叶斯迭代主线**：交替闭式更新后验均值/协方差、噪声精度 \(\lambda\) 与稀疏超参数 \(\gamma_i\)，弱原子自动稀释。
- **TOA恢复**：从后验功率显著的原子提取 \(\tau_i=\theta_i/\Delta f\) 与幅度 \(|a_i|\)，即可获得多径时延估计。

> 参考实现：入口 `run_ofdm_toa.py`，核心算法位于 `sbl/sbl_algorithm.py` 与 `sbl/ofdm_toa.py`。
