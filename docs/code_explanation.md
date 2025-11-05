# 代码实现详解

本文档详细说明仓库中各个模块的作用、核心公式的离散化方法、以及如何运行完整的实验流程复现论文中的算法。

## 总体结构

```
SBL/
├── paper.md                  # 论文原文
├── run_experiment.py         # 实验主脚本
├── sbl/
│   ├── __init__.py           # 包的入口, 暴露主要API
│   ├── atoms.py              # 傅里叶原子及导数
│   ├── metrics.py            # 评估指标
│   ├── sbl_algorithm.py      # 核心SBL推断流程
│   └── simulation.py         # 仿真数据生成
└── docs/
    └── code_explanation.md   # 本说明文档
```

各文件均使用中文注释, 以方便理解。

## 数学背景与实现要点

论文讨论的模型为

\[
\mathbf{y} = \sum_{i=1}^{K} \psi(\theta_i) \alpha_i + \mathbf{w},
\]

其中\(\psi(\theta)\)为参数化的原子。我们以傅里叶原子为例: \(\psi(\theta)[n] = e^{j2\pi\theta n}\)。

- 原子及导数的实现位于`sbl/atoms.py`。通过指数函数直接构造原子, 并给出一阶、二阶导数, 为牛顿法中的梯度/海森矩阵计算提供支撑。
- 评估指标`sbl/metrics.py`实现了归一化均方误差 (Normalized MSE) 以及论文中的频率误差指标\(\beta\)。
- 仿真数据`sbl/simulation.py`按照论文第IV节设定生成频率成对、幅度带噪的测试信号。

核心推断算法位于`sbl/sbl_algorithm.py`, 其思路与论文的算法1一致:

1. **概率模型参数**: `SBLConfig`中配置\(\varepsilon, a, b, \rho\)等超参数。\(\rho=1\)对应复数情形。
2. **矩阵构造**: `_compute_dictionary`返回当前频率集合对应的字典矩阵\(\Psi\)。`_compute_B_inverse`计算\(\mathbf{B} = \mathbf{I} + \Psi\Gamma\Psi^H\)的逆矩阵, 用于后续统计量。
3. **统计量计算**: `_statistics_for_theta`实现附录中的\(s_i, q_i\)及其各阶导数, 以牛顿法为基础。
4. **超参数更新**:
   - `_update_gamma`实现公式(12), 根据当前\(\lambda, s_i, q_i\)求解\(\gamma_i\)。
   - `_gradient_hessian`对应附录中的\(l'(\theta_i), l''(\theta_i)\)。
   - `_newton_refine`执行公式(13)的牛顿迭代, 同时保证频率取值始终在\([0,1)\)。
5. **主循环** (`fit`方法):
   - 每5次迭代执行一次网格搜索, 以\(3N\)个初值激活新原子。
   - 对所有已激活原子, 根据去除自身后的\(\mathbf{B}_{-i}^{-1}\)运行牛顿迭代。
   - 使用公式(11)更新噪声精度\(\lambda\)。
   - 根据频率变化判断收敛。
6. **系数估计**: 收敛后利用公式(4)-(5)计算后验均值\(\hat{\boldsymbol{\alpha}}\)。

## 运行实验

`run_experiment.py`提供了一个可直接运行的脚本, 重现论文中在不同信噪比下重复实验的流程。核心步骤如下:

1. **生成数据**: 调用`generate_line_spectrum`生成满足成对频率约束的信号与噪声观测。
2. **模型推断**: 初始化`SparseBayesianLearning`, 运行`fit`进行SBL推断。
3. **指标统计**: 对重构信号、频率及模型阶数分别计算MSE、\(\beta\)及估计的原子数量。
4. **多次重复**: 对每个SNR执行多次实验并输出平均值。

运行示例:

```bash
python run_experiment.py
```

脚本默认对`[0, 5, 10, 15, 20]` dB进行5次重复实验, 并打印结果。

## 自定义使用

- 若需调整超参数, 可自定义`SBLConfig`, 例如修改`epsilon`或牛顿迭代次数。
- 若有不同信噪比或样本长度, 可修改`LineSpectrumConfig`。
- 若要嵌入其他系统, 可以直接导入`SparseBayesianLearning`类, 将观测向量传入`fit`, 再调用`reconstruct_signal`获取估计结果。
- 对真实OFDM导频数据的TOA提取, 请参考新增的`run_ofdm_toa.py`脚本与`docs/ofdm_toa_explanation.md`说明, 其中给出了与论文模型的对应关系及可视化流程。

本实现保持与论文描述一致的流程, 并辅以中文注释和本说明文档, 便于快速掌握。欢迎在此基础上进一步扩展或优化。
