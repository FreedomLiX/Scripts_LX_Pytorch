import glob
import numpy as np
from utils_anchor.kmeans import kmeans, avg_iou
from tqdm import tqdm

ANNOTATIONS_PATH = "/home/ypp/cheng/datasets/data/labels/data/"
# yolov3-tiny用6，yolov3用9
# CLUSTERS = 9
CLUSTERS = 9
W = 512
H = 512


def load_dataset(path):
    dataset = []
    print("load txt...")
    for txt_file in tqdm(glob.glob("{}/*txt".format(path))):
        # print(txt_file)
        f = open(txt_file, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            l = line.split(' ')
            if float(l[3]) == 0 or float(l[4]) == 0:
                print(txt_file)
                continue
            # print(np.float64(l[0]), np.float64(l[1]))
            dataset.append([np.float64(l[3]), np.float64(l[4])])  # [bbox_w/img_w, bbox_h/img_h]
        # dataset.append([xmax - xmin, ymax - ymin])
        f.close()

    return np.array(dataset)


if __name__ == '__main__':
    data = load_dataset(ANNOTATIONS_PATH)
    print(data[:2])
    out = kmeans(data, k=CLUSTERS)
    anchor_first = list(out[:, 0])
    anchor_second = list(out[:, 1])
    anchor_first.sort()
    anchor_second.sort()
    for i in range(len(anchor_first)):
        print(str(int(anchor_first[i] * W)) + '  ' + str(int(anchor_second[i] * H)))
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    # print("Boxes:\n {}-{}".format(out[:, 0] * W, out[:, 1] * H))

    # ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    # print("Ratios:\n {}".format(sorted(ratios)))
