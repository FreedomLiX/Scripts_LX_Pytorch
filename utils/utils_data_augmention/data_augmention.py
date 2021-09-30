"""
********************************************************
(a) Use Albumentations lib for specific task in Data-Augmentation
(b) Be adequate for Bounding box --yolo--
Date :2021-05
********************************************************
"""

import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from collections import Counter
import copy
from write_xml import write_xml
from tqdm import tqdm


# 定义增强方法
def define_transform():
    transform = A.Compose(
        [
            # 1、 翻转
            A.Sequential([
                A.HorizontalFlip(),
                # A.VerticalFlip()
            ]),
            # 2、噪声、颜色、色彩
            A.OneOf([
                # 噪声
                A.Sequential([
                    A.GaussNoise(),
                    A.ISONoise(),
                    A.MultiplicativeNoise()]),
                # 对比度
                A.Sequential([
                    A.RandomBrightnessContrast(),
                    A.RandomGamma()]),
                # HSV
                A.Sequential([
                    A.HueSaturationValue(),
                    A.CLAHE(),
                    # A.RGBShift(),
                    # A.ColorJitter()
                    ]),
            ], p=0.6),
            # 4、运动模糊
            A.OneOf([
                A.GlassBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.MotionBlur(),
            ], p=0.6),

            # 5、 旋转、放射、切除
            # A.CoarseDropout(p=1),
            # A.RandomSizedCrop(min_max_height=0.3, height=10, width=10),
            A.IAAAffine(),
            A.Rotate(always_apply=False, p=0.8, limit=(-30, 30), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            # A.ShiftScaleRotate(p=1),

            # 6、天气变换 weather transforms
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9,
                             drop_width=1, blur_value=5, p=1),                    # 模拟下雨
                A.RandomSnow(brightness_coeff=1.5,
                             snow_point_lower=0.3, snow_point_upper=0.5, p=1),    # 模拟下雪
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5,
                            alpha_coef=0.1, p=1),                                 # 模拟上雾
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),
                                 angle_lower=0.5, p=1),                           # 模拟太阳
            ], p=0.6),
        ],
        bbox_params=A.BboxParams(format='coco'),
        # min_area=0.0、 min_visibility、滤除变换之后面积小或者变换之后的比例小的bbox drop。
    )
    return transform


def get_labels(xml_path):
    # 获取类别标签对应的ID
    files = os.listdir(xml_path)
    names = []
    for file in files:
        tree = ET.parse(os.path.join(xml_path, file))
        for obj in tree.findall('object'):
            name = obj.find('name').text
            names.append(name)
    result = Counter(names)
    labels = list(result)
    labels.sort()
    return labels


# 获取单个xml文件 所包含的label 以及 budbox
def get_budboxes_labels(xml_file):
    # 由单个xml文件形成数据:
    root = ET.parse(xml_file).getroot()
    budboxes_labels = []
    for object in root.findall('object'):
        # 类别:
        label = object.find('name').text
        # 边框
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        budbox = [xmin, ymin, xmax - xmin, ymax - ymin, label]
        budboxes_labels.append(budbox)
    return budboxes_labels


# 定义输入输出路径、以及增强倍数
def write_jpg_xml(transform, JPG_PATH, XML_PATH, out_path, num=10):
    image_files = os.listdir(JPG_PATH)
    for image_file in tqdm(image_files):
        xml_abspath = os.path.join(XML_PATH, image_file[:-4] + '.xml')
        image_abspath = os.path.join(JPG_PATH, image_file)
        image = cv2.imread(image_abspath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        budboxes_labels = get_budboxes_labels(xml_abspath)
        for i in range(0, num):
            # 增强获取
            transformed = transform(image=image, bboxes=budboxes_labels)
            # 数据本地保存
            write_xml.create_xml(i,
                                 transformed['image'], transformed['bboxes'],
                                 out_path, image_file)


if __name__ == "__main__":
    # 1.配置输入文件路径
    xml_Path = r'C:\PycharmProjects\Utils\datasets\Annotations'
    jpg_Path = r'C:\PycharmProjects\Utils\datasets\JPEGImages'
    out = r'C:\PycharmProjects\Utils\datasets\A'
    # 2. 增强数据的方法
    transform = define_transform()
    # 3. 增强并保存到本地
    write_jpg_xml(transform=transform,                    # 数据增强方法
                  JPG_PATH=jpg_Path, XML_PATH=xml_Path,   # 输入图片和标签文件路径
                  out_path=out,                           # 增强数据保存路径
                  num=10)                                 # 单张图片增强数量
