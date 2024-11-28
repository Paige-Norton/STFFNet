import numpy as np
import cv2

def bilinear_interpolation1(img, scale):
    h, w, _ = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x, y = i / scale, j / scale
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)
            dx, dy = x - x1, y - y1

            new_img[i, j] = (1 - dx) * (1 - dy) * img[x1, y1] + dx * (1 - dy) * img[x2, y1] + \
                            (1 - dx) * dy * img[x1, y2] + dx * dy * img[x2, y2]

    return new_img

def bilinear_interpolation(img, scale):
    c, h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    new_img = np.zeros((3, new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x, y = i / scale, j / scale
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)
            dx, dy = x - x1, y - y1

            for k in range(c):
                new_img[k, i, j] = (1 - dx) * (1 - dy) * img[k, x1, y1] + dx * (1 - dy) * img[k, x2, y1] + \
                                   (1 - dx) * dy * img[k, x1, y2] + dx * dy * img[k, x2, y2]

    return new_img

# 读取图片
img = cv2.imread('/home/ubuntu/BSD/BSD_2ms16ms_all/train/049/Sharp/RGB/00000040.png')

# 指定放大倍数
scale = 0.5

# 进行双线性插值
new_img = bilinear_interpolation1(img, scale)

# 显示原始图片和放大后的图片
cv2.imwrite('Original Image.jpg', img)
cv2.imwrite('Bilinear Interpolated Image.jpg', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
