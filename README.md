python run_ofdm_toa.py \
    --input-path ./your_real_data.npy \
    --subcarrier-spacing-hz 12000 \
    --num-samples 480 \
    --save-path ./toa_result.png



# === 每个样本写两行：第一行TOA(ns)，第二行对应幅度，并在第一列加入样本编号 ===
csv_path = Path("toa_log.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

# 取出 TOA 和 幅度
toas = np.asarray(result.toas, dtype=float) * 1e9  # 转ns
amps = np.abs(np.asarray(result.amplitudes, dtype=complex))  # 取幅度模值

# 转成 list 便于写入
toas_list = toas.tolist()
amps_list = amps.tolist()

with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([idx] + toas_list)  # 第1行：样本编号 + TOA
    writer.writerow([idx] + amps_list)  # 第2行：样本编号 + 幅度
# === 结束 ===
