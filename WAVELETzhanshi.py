import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

#图像信号的低频部分（低通带）表示图像的基本信息（平滑信息），而高频部分（高通带）表示图像的细节信息。就好比我们用谷歌地图一样，（低频）就意味着没有细节，是一个整体的视图；高频）意味着更多的细节信息。
#每次小波变换后，图像便分解为4个大小为原来尺寸1/4的子块区域，分别包含了相应频带的小波系数，相当于在水平方向和坚直方向上进行隔点采样。进行下一层小波变换时，变换数据集中在频带上。
#对图像每进行一次小波变换，会分解产生一个低频子带（LL：行低频、列低频）和三个高频子带(垂直子带LH：行低频、列高频；水平子带HL：行高频、列低频；对角子带HH：行高频、列高频）
# 后续小波变换基于上一级低频子带LL进行，依次重复，可完成对图像的k级小波变换
#小波变换是可逆的，进行小波分解得到的子图可通过组合重构原图
def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

    

dwt_module=DWT()
x=Image.open("/home/ubuntu/BSD/BSD_2ms16ms/train/024/Blur/RGB/00000087.png")
# x=Image.open('./mountain.png')
x=transforms.ToTensor()(x)
# print(x)
x=torch.unsqueeze(x,0)
x=transforms.Resize(size=(256,256))(x)
subbands=dwt_module(x)

title=['LL','HL','LH','HH']

plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    temp = subbands[0, 3*i:3*(i+1), :, :].permute(1, 2, 0)
# 将图像数据限制在 [0, 1] 范围内
    temp = np.clip(temp, 0, 1)
    print(temp.shape)  # 打印 temp 的形状

    plt.imshow(temp)
    plt.title(title[i])
    plt.axis('off')
plt.savefig('/home/ubuntu/ESTRNN_Trans/sample_xiaobo_split.png')
plt.show()


title=['Original Image','Reconstruction Image']
reconstruction_img=IWT()(subbands).cpu()
show_list = [x[0].permute(1, 2, 0), reconstruction_img[0].permute(1, 2, 0)]

plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(show_list[i])
    plt.title(title[i])
    plt.axis('off')
plt.savefig('/home/ubuntu/ESTRNN_Trans/sample_xiaobo_img.png')
plt.show()