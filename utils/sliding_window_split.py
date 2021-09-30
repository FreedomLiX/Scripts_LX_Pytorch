"""
Sliding Window Split Images
"""
import cv2
from imutils import paths
import os


def sliding_window_split(images_Path, save_Path, kernelSize=250, stride=100):
    # 1：读取数据
    images_paths = list(paths.list_images(images_Path))
    for index1, filePath in enumerate(images_paths):
        # 2: 图像路径、和图像名
        # filename = filePath.split(os.path.altsep)[-1]  # 注意windows 和 linux 系统 字符切分的方式
        label = filePath.split(os.path.altsep)[-2]
        # 3：图像切分
        img = cv2.imread(filePath)
        if img is not None:
            Height, Width, _ = img.shape
            # 4：切分
            cuts = []
            # temp = digits(1 + (i - 1) * b:i * b, 1 + (j - 1) * b: j * b);
            for i in range(0, Width, stride):
                if i + kernelSize >= Width:
                    break
                for j in range(0, Height, stride):
                    if j + kernelSize >= Height:
                        break
                    cut = img[j:j + kernelSize, i:i + kernelSize, :]
                    print(cut.shape)
                    cuts.append(cut)
            # 保存到本地
            for index2, cut in enumerate(cuts):
                #     cv2.imshow('win', cut)
                cv2.imwrite(save_Path + label + str(index1) + str(index2) + '.jpg', cut)
        # cv2.imencode('.jpg', cut)[1].tofile(save_Path + label + str(index1) + str(index2) + '.jpg')
        else:
            continue


if __name__ == '__main__':
    # 脚本使用说明：
    # 1：指定图片文件具体位置（输入输出，文件）
    Images_Path = './Fire_Neutral_Smoke/Fire/'
    Out_Path = './Fire_Neutral_Smoke/FireOut/'
    # 2：调用函数
    sliding_window_split(images_Path=Images_Path, save_Path=Out_Path)
