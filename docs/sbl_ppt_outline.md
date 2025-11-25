# SBL频域提取TOA汇报PPT要点

以下为1-2页PPT的核心内容，可直接复制到幻灯片中。

## 第1页：问题背景与模型假设
- **任务**：从OFDM导频最小二乘频响估计多径到达时间(TOA)。
- **观测模型**：频域响应\(H[k]=\sum_{i=1}^K a_i e^{-j2\pi f_k \tau_i}+w[k]\)，取共轭后与论文中的指数字典完全对应。\(f_k=k\,\Delta f\)。
- **字典原子**：傅里叶原子\(\psi(\theta)[k]=e^{j2\pi\theta k}\)，其中\(\theta_i=\Delta f\,\tau_i\)。
- **关键假设**：路径稀疏、噪声为复高斯，且\(\Delta f\,\tau_i<1\)以避免循环前缀模糊。

## 第2页：SBL核心流程与输出
- **初始化**：
  - 频率网格\(\theta\)覆盖\([0,1)\)或结合先验可缩窄；
  - 设定先验方差\(\gamma\)为小正值、噪声精度\(\lambda\)为1，迭代上限与稀疏阈值由`SBLConfig`控制。
- **迭代估计**(对应`sbl.sbl_algorithm.SparseBayesianLearning.fit`):
  1. 计算后验协方差\(\Sigma=(\lambda\Psi^\mathrm{H}\Psi+\Gamma^{-1})^{-1}\)与均值\(\mu=\lambda\Sigma\Psi^\mathrm{H}y\)。
  2. 依据后验功率\(|\mu_i|^2+\Sigma_{ii}\)更新每个路径的先验尺度\(\gamma_i\)。
  3. 用残差能量更新噪声精度\(\lambda\)，若\(\gamma\)或\(\lambda\)收敛则停止。
- **网格微调**：对激活原子执行牛顿步长修正\(\theta\leftarrow\theta-\frac{\partial\mathcal{L}/\partial\theta}{\partial^2\mathcal{L}/\partial\theta^2}\)，提升延迟估计精度。
- **TOA恢复与筛选**：
  - 将收敛的\(\theta\)除以\(\Delta f\)得到TOA；幅度取后验均值\(|\mu|\)。
  - 可按\(\gamma\)或幅度阈值剔除弱路径，保留主要多径。
- **可视化**：按快照编号绘制TOA散点，横轴为`num`，纵轴为连续时间坐标(如µs)，颜色可映射幅度大小，直观呈现多径随时间演化。

> 参考实现：`run_ofdm_toa.py`负责数据入口与绘图，算法细节集中在`sbl/sbl_algorithm.py`与`sbl/ofdm_toa.py`。
