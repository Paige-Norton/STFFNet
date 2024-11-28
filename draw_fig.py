import matplotlib.pyplot as plt
import re

train_psnr_bistlm_differ = []
valid_psnr_bistlm_differ = []
train_psnr_bistlm_fu = []
valid_psnr_bistlm_fu = []
epochs = list(range(1, 200+1))  # 假设有 200 个 epochs

#
file_path_bistlm_differ = "/home/ubuntu/ESTRNN_Trans/experiment/2024_03_31_15_16_26_ESTRNN_BSD_halfdata_epoch200_newbackbone_傅里叶_图片相减/log.txt"
with open(file_path_bistlm_differ, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "[train]" in line and "PSNR" in line:
            psnr_match = re.search(r'PSNR : (\d+\.\d+)', line)
            if psnr_match:
                psnr_value = float(psnr_match.group(1))
                train_psnr_bistlm_differ.append(psnr_value)
        elif "[valid]" in line and "PSNR" in line:
            psnr_match = re.search(r'PSNR : (\d+\.\d+)', line)
            if psnr_match:
                psnr_value = float(psnr_match.group(1))
                valid_psnr_bistlm_differ.append(psnr_value)

file_path_bistlm_fu = "/home/ubuntu/ESTRNN_Trans/experiment/2024_01_13_06_35_25_ESTRNN_BSD_halfdata_epoch200_newbackbone_傅里叶/log.txt"
with open(file_path_bistlm_fu, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "[train]" in line and "PSNR" in line:
            psnr_match = re.search(r'PSNR : (\d+\.\d+)', line)
            if psnr_match:
                psnr_value = float(psnr_match.group(1))
                train_psnr_bistlm_fu.append(psnr_value)
        elif "[valid]" in line and "PSNR" in line:
            psnr_match = re.search(r'PSNR : (\d+\.\d+)', line)
            if psnr_match:
                psnr_value = float(psnr_match.group(1))
                valid_psnr_bistlm_fu.append(psnr_value)

# 绘制 PSNR 变化图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_psnr_bistlm_differ, label='Train PSNR')
plt.plot(epochs, valid_psnr_bistlm_differ, label='Valid PSNR')
plt.plot(epochs, train_psnr_bistlm_fu, label='Train PSNR')
plt.plot(epochs, valid_psnr_bistlm_fu, label='Valid PSNR')

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR Changes Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("savefig")
plt.show()
