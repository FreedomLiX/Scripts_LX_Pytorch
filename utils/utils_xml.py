"""
utils_xml.py：
1、基于xml和jpg数据，进行数据集的统计、处理；
2、基于数据集“labels”分布，进行数据集等百分比切分
"""
import os
import shutil
from xml.dom import minidom

import numpy as np
from collections import Counter
from tqdm import tqdm
import xml.etree.ElementTree as ET
import glob
from utils_anchor.kmeans import kmeans, avg_iou
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from utils_plot import utils_plot
from utils_budbox import utils_bndbox
from PIL import Image
import os.path
import xml.dom.minidom


def get_labels(xml_Path):
    # 获取xml文件标签名，并写到./cfg/classnames.names文件内；
    ret = statistic_labels_number(xml_Path=xml_Path, save_plot=False)
    if os.path.exists('./cfg/classnames.names'):
        os.remove('./cfg/classnames.names')
    classnames = list(ret)
    classnames.sort()  # 从新排序确保names顺序唯一
    with open('./cfg/classnames.names', 'w') as file:
        file.write(str(classnames))
    print("classnames:{}".format(classnames))
    return classnames


def statistic_labels_number(xml_Path, save_plot=False):
    """
    xml_path: xml文件路径
    :return: ret 返回统计个数的字典数据类型
    """
    files = os.listdir(xml_Path)
    names = []
    print("【INFO】：loading xml files...")
    for file in tqdm(files):
        tree = ET.parse(os.path.join(xml_Path, file))
        for obj in tree.findall('object'):
            name = obj.find('name').text
            names.append(name)
    print("【INFO】：statisticing different label distribution...")
    result = Counter(names)
    utils_plot.plot_labels_statistic(result, save_plot)
    print(result)
    return result


def statistic_labels_wh(xml_Path, plot_w_h=True, plot_aspect_radio=True):
    # 统计不同标签的宽高、以及宽高比；
    files = os.listdir(xml_Path)
    labels = []
    boxes = []
    for file in tqdm(files):
        tree = ET.parse(xml_Path + file)
        root = tree.getroot()
        size = root.find('size')
        Width = int(size.find('width').text)
        Height = int(size.find('height').text)
        for obj in tree.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            box_xyxy = [float(bndbox.find('xmin').text) / Width, float(bndbox.find('ymin').text) / Height,
                        float(bndbox.find('xmax').text) / Width, float(bndbox.find('ymax').text) / Height]
            # print(box_xyxy)
            box_xywh = utils_bndbox.xyxy2xywh_opencv(box_xyxy)
            # print(box_xywh)
            labels.append(name)
            boxes.append(box_xywh)
    if plot_w_h:
        utils_plot.plot_labels_bndbox(labels, boxes)
    if plot_aspect_radio:
        utils_plot.plot_labels_aspect_ratio(labels, boxes)
    return labels, boxes


def statistic_anchor_cluster(txt_files_path, W=608, H=608, CLUSTERS=9):
    # 依据xml文件，统计锚定框的P3、P4、P5对应的值；
    # anchor boxes：yolov5 和yolov3 训练时，在配置文件设定
    # anchor boxes: yolov5 和yolov3 推理时，与训练保持相同
    """
    :param txt_file_path: txt 文件路径：
    :param W: 训练时，输入图像宽度
    :param H: 训练时，输入图像高度
    :param CLUSTERS: 聚类数量 9类
                      - [3,  10,  5 , 20, 8,  24]     # P3/8
                      - [8,  34, 12,  39, 15,  40]    # P4/16
                      - [25,  67, 33,  81, 53,  155]  # P5/32
    :return:
    """

    def get_w_h(TXT_files_path):
        w_h = []
        for file in tqdm(glob.glob("{}/*txt".format(TXT_files_path))):
            f = open(file, 'r')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                label_x_y_w_h = line.split(' ')
                if float(label_x_y_w_h[3]) == 0 or float(label_x_y_w_h[4]) == 0:
                    print(file)
                    continue
                # print(np.float64(l[0]), np.float64(l[1]))
                w_h.append([np.float64(label_x_y_w_h[3]), np.float64(label_x_y_w_h[4])])
                # [bbox_w/img_w, bbox_h/img_h]
            # w_h.append([xmax - xmin, ymax - ymin])
            f.close()
        return np.array(w_h)

    # kmeans 聚类：
    data = get_w_h(txt_files_path)
    out = kmeans(data, k=CLUSTERS)
    anchor_w = (out[:, 0] * W).astype(np.int16)
    anchor_h = (out[:, 1] * H).astype(np.int16)
    anchor_W = np.sort(anchor_w)
    anchor_H = np.sort(anchor_h)
    # 聚类结果：
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    P3 = anchor_W[0:3], anchor_H[0:3]
    P4 = anchor_W[3:6], anchor_H[3:6]
    P5 = anchor_W[6:9], anchor_H[6:9]
    # 输出对应格式：w h 一一对应；
    print("P3:{},\n P4:{},\n P5:{}".format(P3, P4, P5))
    return P3, P4, P5


def xml_to_txt(xml_Path, txt_Path):
    """
    一个xml生成一个txt
     xml file to txt file :一一对应 one-to-one mapping
    """

    def Standardization_xywh(IMAGE_WH, box):
        """
        :标准化，x,y,w,h
        """
        dw, dh = 1. / (IMAGE_WH[0]), 1.0 / (IMAGE_WH[1])
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1.0, (box[2] + box[3]) / 2.0 - 1.0, \
                     abs(box[1] - box[0]), abs(box[3] - box[2])
        x, y = x * dw, y * dh
        w, h = w * dw, h * dh
        x1 = max(0, x)
        x1 = min(1, x)
        y1 = max(0, y)
        y1 = min(1, y)
        h1 = max(0, h)
        h1 = min(1, h)
        w1 = max(0, w)
        w1 = min(1, w)
        if x != x1 or y != y1 or w != w1 or h != h1:
            return [x1, y1, w1, h1], True
        return [x, y, w, h], False

    # from xml to names and Standardization_xywh txt file:
    classnames = list(get_labels(xml_Path))  # labels的顺序问题，确保唯一
    fileslist = os.listdir(xml_Path)
    print("【INFO】：xml to txt ...")
    for i in tqdm(range(0, len(fileslist))):
        path = os.path.join(xml_Path, fileslist[i])
        if ('.xml' in path) or ('.XML' in path):
            with open(path, "r") as in_file:
                txtname = fileslist[i][:-4] + '.txt'
                txtpath = txt_Path
                if not os.path.exists(txtpath):
                    os.makedirs(txtpath)
                txtfile = os.path.join(txtpath, txtname)
                if os.path.exists(txtfile):
                    os.remove(txtfile)
                with open(txtfile, "w+") as out_file:
                    tree = ET.parse(in_file)
                    root = tree.getroot()
                    size = root.find('size')
                    w = int(size.find('width').text)
                    h = int(size.find('height').text)
                    out_file.truncate()
                    for obj in root.iter('object'):
                        difficult = obj.find('difficult').text
                        cls = obj.find('name').text
                        if cls not in classnames or int(difficult) == 1:
                            continue
                        cls_id = classnames.index(cls)
                        xmlbox = obj.find('bndbox')
                        b = [float(xmlbox.find('xmin').text),
                             float(xmlbox.find('xmax').text),
                             float(xmlbox.find('ymin').text),
                             float(xmlbox.find('ymax').text)]
                        bb, out_of_bounds = Standardization_xywh((w, h), b)
                        if out_of_bounds:
                            if b[0] < 0 or b[0] > w:
                                b[0] = max(b[0], 0)
                                b[0] = min(b[0], w)
                                xmlbox.find('xmin').text = str(b[0])
                            if b[1] < 0 or b[1] > w:
                                b[1] = max(b[1], 0)
                                b[1] = min(b[1], w)
                                xmlbox.find('xmax').text = str(b[1])
                            if b[2] < 0 or b[2] > h:
                                b[2] = max(b[2], 0)
                                b[2] = min(b[2], h)
                                xmlbox.find('ymin').text = str(b[2])
                            if b[3] < 0 or b[3] > h:
                                b[3] = max(b[3], 0)
                                b[3] = min(b[3], h)
                                xmlbox.find('ymax').text = str(b[3])
                            tree.write(path)
                            print("out_of_bounds:\t", fileslist[i])
                        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        else:
            print("please check your xml file...")
    print("【INFO】：xml to txt SUCCESS！")


def get_xmlFiles_different_labels(xml_Path, save=False):
    """ 遍历xml文件，统计不同labels的数量，以及labels的总和；
        构建一个表格：
        表头为：文件名，label1、label2、label3、label4...所有标签，以及本文件标签之和Sum
    """
    # 1、构建表头：
    LABELS = get_labels(xml_Path)
    labels = get_labels(xml_Path)
    labels.append('Sum')
    labels.insert(0, 'filename')
    # 2、构建空表
    # 3、填充表格
    filenamess = os.listdir(xml_Path)
    data_ = pd.DataFrame(np.zeros((len(filenamess), len(labels)), dtype=int), columns=labels)
    for index, current_filename in enumerate(filenamess):
        # 填充一行，构建一行的表格
        data_.loc[index, 'filename'] = current_filename.replace('.xml', '')
        for label in LABELS:
            # 解析当前xml文件，统计不同labels的分布
            tree = ET.parse(os.path.join(xml_Path, current_filename))  # 解析文件
            for obj in tree.findall('object'):
                current_xml_label = obj.find('name').text
                if label == current_xml_label:
                    data_.loc[index, label] += 1
        data_.loc[index, 'Sum'] = np.sum(data_.iloc[index, 1:])
    if save:
        data_.to_csv("data.csv", index=False)
    return data_


def train_and_test_datasets_txtfile_generator(xml_Path):
    """
    数据集等百分比切分，并生成train.txt,test.txt:图片的路径
    """

    # 数据格式text的数据格式：
    def write_to_txt(data, txtfile, pre_path='./data'):
        if os.path.exists(txtfile):
            os.remove(txtfile)
        with open(txtfile, 'w') as out_file:
            for item in data:
                out_file.write(pre_path + '/' + item + '.jpg\n')

    # 1、获取数据表格：
    data_sheet = get_xmlFiles_different_labels(xml_Path)
    # 2、判断数据表格Sum和是否满足“等百粉比”group数量至少为1的要求，参考链接：
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    # 3、对只有1的 group 图片的数据进行去除处理（不参与训练集、测试集的生成）
    counter_dic = Counter(data_sheet['Sum'])
    for labels, groups in counter_dic.items():
        if groups == 1:  # 含有n个标签的图片只有一张
            # data_sheet['Sum'].where(data_sheet['Sum'] > int(labels), 2, inplace=True)
            # 定位数量只有1的 group 的行索引
            index = data_sheet[(data_sheet['Sum'] == int(labels))].index
            # 去除数量只有1的 group 的索引
            data_sheet.drop(index, inplace=True)
    data_sheet.to_csv('./utils_xml/data.csv')
    DATA_SHEET = pd.read_csv('./utils_xml/data.csv')
    # 4、等百分比分布数据切分
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1971587)
    TRAIN, TEST = [], []
    for train_index, test_index in split.split(DATA_SHEET, DATA_SHEET['Sum']):
        TRAIN = list(train_index)
        TEST = list(test_index)
    # 5、索引filename列，进行索引提取，文件
    train = list(DATA_SHEET['filename'][TRAIN])
    test = list(DATA_SHEET['filename'][TEST])
    # 5、保存到本地txt文件中
    write_to_txt(train, "train.txt")
    write_to_txt(test, "test.txt")
    print("【INFO】：train.txt and test.txt have been generated SUCCESS！")


def check_WH(xmlpath, imgpath):
    """
    检查图宽高是否正确
    """
    def convert_annotation(xmlpath, imgpath, xmlname):
        xmlfile = os.path.join(xmlpath, xmlname)
        imgfile = os.path.join(imgpath, xmlname[:-4] + '.jpg')
        img = Image.open(imgfile)
        W, H = img.size
        with open(xmlfile, "r") as in_file:
            tree = ET.parse(in_file)
            size = tree.getroot().find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if (W != w) or (H != h):
                print("XML width:{} or height:{}  is not same as image's".format(w, h))
                size.find('height').text = str(H)
                size.find('width').text = str(W)
                print("XML width and height has been changed")
            tree.write(xmlfile)

    list = os.listdir(xmlpath)
    print("【INFO】: Checking XML width and height is same as image's or not!!!")
    for i in tqdm(range(0, len(list))):
        path = os.path.join(xmlpath, list[i])
        if ('.xml' in path) or ('.XML' in path):
            convert_annotation(xmlpath, imgpath, list[i])
        else:
            print('not xml file', i)
        pass


def rename_label(xml_Path, old_name, new_name):
    print("【INFO】: renaming label ...")
    files = os.listdir(xml_Path)  # 得到文件夹下所有r文件名称
    for xmlFile in tqdm(files):
        if not os.path.isdir(xmlFile):
            print(xmlFile)
        dom = xml.dom.minidom.parse(os.path.join(xml_Path, xmlFile))
        root = dom.documentElement
        name = root.getElementsByTagName('name')
        for i in range(len(name)):
            if name[i].firstChild.data == old_name:  # 节点的旧名字
                name[i].firstChild.data = new_name
                print(name[i].firstChild.data)
            # 保存修改到xml文件中
        with open(os.path.join(xml_Path, xmlFile), 'w', encoding='UTF-8') as fh:
            dom.writexml(fh)
            print("【INFO】: rename label Done!")


def delete_label(xml_path, label):
    print("【INFO】: deleting label...")
    for _, _, filenames in tqdm(os.walk(xml_path)):  # os.walk遍历目录名
        for filename in filenames:
            if filename.endswith('.xml'):
                tree = ET.parse(os.path.join(xml_path, filename))
                root = tree.getroot()
                for object in root.findall('object'):  # 找到根节点下所有“object”节点
                    name = str(object.find('name').text)  # 找到object节点下name子节点的值（字符串）
                    # 如果name等于str，则删除该节点
                    if name in label:
                        root.remove(object)
                        print("【INFO】：removed {}".format(name))
                tree.write(os.path.join(xml_path, filename))

def find_label(xml_path, jpg_path, label, out_path):
    """
    # 查找含有 “某个或某些” 类别标签，寻找xml文件
    :param xml_path:
    :param label:  ['Car','Bus']
    :param out_path:
    :return:
    """
    out_xml_path = os.path.join(out_path, 'Annotations')
    out_jpg_path = os.path.join(out_path, 'JPEGImages')
    if not os.path.exists(out_xml_path):
        os.makedirs(out_xml_path)
    if not os.path.exists(out_jpg_path):
        os.makedirs(out_jpg_path)

    for filename in os.listdir(xml_path):
        xml_file_path = os.path.join(xml_path, filename)
        jpg_file_path = os.path.join(jpg_path, str(filename[:-4])+str('.jpg'))
        dom = minidom.parse(xml_file_path)
        collection = dom.documentElement
        objects = collection.getElementsByTagName("object")
        for obj in objects:
            name = obj.getElementsByTagName('name')[0]
            name_data = name.childNodes[0].data
            if name_data in label:
                print("INFO: It's copying {}".format(filename))
                shutil.copy2(xml_file_path, out_xml_path)
                shutil.copy2(jpg_file_path, out_jpg_path)
                break


if __name__ == "__main__":
    xml = '/media/dixn/李想/已标注车辆多维度特征数据_白天近景/Annotations'
    jpg = '/media/dixn/李想/已标注车辆多维度特征数据_白天近景/JPEGImages'
    out = '/media/dixn/李想/已标注车辆多维度特征数据_白天近景/cut'
    find_label(xml, jpg, 'carFace', out)
