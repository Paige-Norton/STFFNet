# import torch
# import torch.nn as nn

# # 定义一个线性层
# linear_layer = nn.Linear(in_features=3, out_features=2)

# # 生成一个二维输入矩阵（假设有两个样本，每个样本具有3个特征）
# input_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# # 对输入进行线性连接
# linear_output = linear_layer(input_matrix)

# # 打印输出
# print("输入矩阵:")
# print(input_matrix)
# print("\n线性连接后的输出:")
# print(linear_output)


# import torch
# import torchvision.transforms as transforms
# import cv2
import matplotlib.pyplot as plt

# # 读取图像
# image_path = "/home/ubuntu/BSD/BSD_2ms16ms/train/055/Blur/RGB/00000040.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 转换为 PyTorch 的 Tensor
# image_tensor = transforms.ToTensor()(image).unsqueeze(0)

# # 使用 Canny 边缘检测
# canny_edges = torch.tensor(cv2.Canny(image, 100, 200) / 255.0).unsqueeze(0).unsqueeze(0).float()

# canny_np = canny_edges.squeeze().numpy()

# # 显示 Canny 边缘检测的结果
# plt.imshow(canny_np, cmap='gray')
# plt.axis('off')
# plt.show()

# # 保存结果为图片
# plt.imsave('canny_edges_result.jpg', canny_np, cmap='gray')


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

# 读取图像
image_path = "/home/ubuntu/BSD/BSD_2ms16ms/train/055/Blur/RGB/00000040.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 转换为 PyTorch 的 Tensor
image_tensor = transforms.ToTensor()(image).unsqueeze(0).float()

# 定义拉普拉斯核
laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
laplacian_kernel = laplacian_kernel.float()
# 使用卷积进行拉普拉斯滤波
laplacian_result = F.conv2d(image_tensor, laplacian_kernel.unsqueeze(0).unsqueeze(0))

canny_np = laplacian_result.squeeze().numpy()

# 显示 Canny 边缘检测的结果
plt.imshow(canny_np, cmap='gray')
plt.axis('off')
plt.show()

# 保存结果为图片
plt.imsave('canny_edges_result.jpg', canny_np, cmap='gray')

