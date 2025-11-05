# OFDM多径TOA提取流程说明

本文档结合论文中的稀疏贝叶斯学习(SBL)框架, 说明如何在OFDM导频的最小二乘频率响应数据上提取多径到达时间(Time of Arrival, TOA)。

## 数据假设

- 输入数据形状为`(num, 1, symbol, att, 480)`:
  - `num`: 采样快照数量, 例如在不同时间或位置采集的样本;
  - 第二维固定为1, 可忽略;
  - `symbol`: 不同OFDM符号的索引;
  - `att`: 接收天线编号;
  - `480`: 子载波数量, 频率间隔固定为Δf=120 kHz。
- 数据由`numpy.load`读取, 每个元素为复数, 表示经最小二乘估计得到的频域导频响应。

## 与论文模型的对应关系

论文中的观测模型写作

\[
\mathbf{y} = \sum_{i=1}^{K} \psi(\theta_i)\alpha_i + \mathbf{w},
\]

其中字典原子\(\psi(\theta)[n] = e^{j2\pi\theta n}\)。在OFDM频域中, 频响可表示为

\[
H[k] = \sum_{i=1}^{K} a_i e^{-j2\pi f_k \tau_i} + w[k], \quad f_k = k\,\Delta f.
\]

只需将数据取共轭, 即可得到与论文相同的指数形式:

\[
H[k]^* = \sum_{i=1}^{K} a_i^* e^{j2\pi (\Delta f \tau_i) k} + w[k]^*.
\]

因此, 原论文中的参数\(\theta_i\)与TOA满足\(\theta_i = \Delta f\,\tau_i\)。

- `sbl.ofdm_toa._prepare_observation`负责取共轭并展平数据。
- `SparseBayesianLearning.fit`保持不变, 直接对上述形式进行推断。
- `SnapshotTOAResult`在后处理阶段将估计的\(\theta_i\)除以Δf, 得到物理含义明确的TOA(单位: 秒)。

## 代码流程

完整的工程流程由`run_ofdm_toa.py`脚本驱动:

1. **加载数据**: 通过`numpy.load`读取`.npy`文件, 并验证张量形状是否符合约定。
2. **配置SBL**: 使用`SBLConfig`设置ε等超参数, 可在命令行中调整稀疏程度与迭代上限。
3. **逐快照估计** (`estimate_toas_for_dataset`):
   - 对每个`num`索引取指定符号、天线的频率响应;
   - `_extract_snapshot_toa`将数据送入`SparseBayesianLearning`, 获得激活的原子集合;
   - 自动完成\(\theta\to\tau\)转换, 并收集后验幅度`|α|`与先验尺度`γ`。
4. **路径筛选** (`filter_paths_by_gamma`): 可选步骤, 利用γ阈值剔除能量较低的虚警路径。
5. **结果可视化**: `_plot_toa`按快照索引绘制散点图, 横轴为`num`, 纵轴为TOA(支持`s/ms/us/ns`单位), 颜色表示后验幅度大小, 有助于观察多径随时间的变化。

下图展示了流程中关键函数之间的关系:

```
run_ofdm_toa.py
├── numpy.load -> 读取频响数据
├── estimate_toas_for_dataset -> sbl.ofdm_toa
│   ├── _extract_snapshot_toa -> SparseBayesianLearning.fit
│   └── SnapshotTOAResult -> θ转换为TOA
├── filter_paths_by_gamma (可选)
└── _plot_toa -> Matplotlib散点图
```

## 使用说明

```bash
python run_ofdm_toa.py data.npy --subcarrier-spacing 120000 \
    --symbol-index 0 --antenna-index 0 --time-unit us --show
```

- `--gamma-threshold`可设置为正数以去除弱路径。
- 若需要保存图像, 可添加`--save-path toa.png`。
- SBL假设TOA满足\(\Delta f\,\tau < 1\), 即小于OFDM循环前缀所对应的最大无模糊延迟; 若超出该范围, 需对输入数据做预处理(例如延迟对齐或扩大频率采样范围)。

通过以上流程, 便可在真实的OFDM导频数据上复现论文算法并获得清晰的TOA可视化结果。
