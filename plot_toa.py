import csv
import numpy as np
import matplotlib.pyplot as plt

csv_file = "toa_log.csv"  # 你的CSV路径

x = []  # 样本编号
y = []  # TOA(ns)
c = []  # 幅度（用于颜色）

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

# 每两行一组：第1行TOA，第2行幅度
for i in range(0, len(rows), 2):
    # 解析两行
    row_toa = rows[i]      # 例如: [idx, t1, t2, t3...]
    row_amp = rows[i + 1]  # 例如: [idx, a1, a2, a3...]

    idx = int(row_toa[0])  # 样本编号
    toas = [float(v) for v in row_toa[1:]]
    amps = [float(v) for v in row_amp[1:]]

    # 逐个多径加入散点数据
    for t, a in zip(toas, amps):
        x.append(idx)
        y.append(t)
        c.append(a)

# 绘制散点
plt.figure(figsize=(10, 6))

# 计算幅度范围并打印（用于调试）
c_array = np.array(c)
vmin, vmax = c_array.min(), c_array.max()
print(f"Amplitude range: [{vmin:.6f}, {vmax:.6f}]")

# 设置颜色映射范围（使用小点避免覆盖）
sc = plt.scatter(x, y, c=c, s=3, cmap="jet", vmin=vmin, vmax=vmax, marker='.')
plt.colorbar(sc, label="Amplitude")

plt.xlabel("Sample Index")
plt.ylabel("TOA (ns)")
plt.title("Multi-path TOA Scatter (Color = Amplitude)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()
