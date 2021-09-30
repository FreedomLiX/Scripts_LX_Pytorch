import os

from imgaug import augmenters as iaa
import cv2

seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
    # iaa.Crop(px=(0, 16)),           # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
    iaa.Fliplr(0.5),                # 对百分之五十的图像进行做左右翻转
    iaa.OneOf([iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
               iaa.Multiply((0.8, 1.5)),
               iaa.MultiplyHue(),
               iaa.Affine(rotate=(-3, 3)),
               iaa.AddToHueAndSaturation((-30, 30)),
               iaa.Resize(112),
               iaa.Resize(448),
               iaa.Rain(),
               iaa.MedianBlur()])
])
in_path = '/media/dixn/李想/res/091631147jsm_rename/VRID/gallery'
out_path = '/media/dixn/李想/res/091631147jsm_rename/VRID/out/'
num = 5
for file in os.listdir(in_path):
    image = cv2.imread(os.path.join(in_path, file))
    for i in range(num):
        images_aug = seq.augment_image(image)  # 单张图片的数据增强方式。
        # cv2.imshow('0', images_aug)
        cv2.imwrite(os.path.join(out_path, file.split('_')[0] + '_' + str(i) + '_' + file.split('_')[1]), images_aug)
