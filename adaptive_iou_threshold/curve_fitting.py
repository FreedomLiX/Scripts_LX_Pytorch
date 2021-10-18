"""
curve fitting
step 1 : down Envelope data
step 2 : curve fitting
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scipy import signal


class CurveFitting(object):
    def __init__(self, txt_path):
        super(CurveFitting, self).__init__()
        self.txt = txt_path

    def __get_data(self):
        assert os.path.exists(self.txt)
        Data = []
        with open(self.txt, 'r') as file:
            file_lines = file.readlines()
            for data in file_lines:
                data = eval(data.strip('\n').split('\t')[1])
                Data.append(data)
        IOUS = [self.get_iou(box1, box2) for box1, box2 in zip(Data[1:], Data[:-1])]
        return IOUS

    @staticmethod
    def get_iou(box1, box2):
        """
        [x,y,w,h] ==> [xmin1, ymin1, xmax1, ymax1]
        """
        xmin1, ymin1, xmax1, ymax1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        xmin2, ymin2, xmax2, ymax2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        # 计算每个矩形的面积
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
        # 计算相交矩形
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        a1 = w * h  # C∩G的面积
        a2 = s1 + s2 - a1
        iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
        if iou <0.5:
            iou = 0.5
        return iou

    def get_down_peaks(self, save_plot=False):
        data = self.__get_data()
        print(len(data))
        x = np.linspace(0, 1920, len(data)).reshape(-1, 1)
        data = np.asarray(data).reshape(-1, 1)

        data_min = np.abs(data.reshape(1, -1)[0] - 1)
        peaks, _ = signal.find_peaks(data_min, distance=5)
        min_data = data[peaks]
        # 拟合结果
        x_, y_ = self.plot_logistic_dis()
        # print(x_.shape, y_.shape)
        if save_plot:
            plt.title("Iou-distribution")
            plt.grid()
            plt.xlabel("pixel x-coordinate value")
            plt.ylabel("iou")
            plt.plot(x, data, label='iou')
            plt.plot(x[peaks], min_data, "x", label="local peaks")
            # plt.plot(x, y_, label="curve-fitting")
            plt.legend()

            plt.savefig(".\iou_data\peaks.png")
        return peaks, x[peaks], min_data

    # def get_smoothing(self, x, data):
    #     from scipy.signal import savgol_filter
    #     data = savgol_filter(np.array(data), 51, 3, mode="wrap")
    #     plt.plot(data, label='iou')
    #     plt.plot(peaks, min_data, "x", label="local peaks")

    # def plot_data(self, plot=True):
    #
    #     x = np.linspace(0, 1920, len(data)).reshape(-1, 1)
    #
    #     # hx = signal.hilbert(data)
    #     # amplitude_envelope = np.abs(hx)
    #
    #     data_smoothing = self.smoothing(peaks, min_data)
    #
    #     # x, y3, pred_data = self.get_curve(x, min_data)
    #     # y_curve = self.curve_fitting(x, y3)
    #     # print(data[peaks])
    #     if plot:
    #         plt.title("Iou data distribution")
    #         plt.grid()
    #         plt.xlabel("pixel x-coordinate value")
    #         plt.ylabel("iou threshold")
    #         plt.plot(data, label="IOU")
    #         plt.plot(peaks, data_smoothing, label="data_smoothing")
    #         # plt.plot(peaks, pred_data, label="Pred-Curve")
    #         # plt.plot(x, amplitude_envelope, label="envelope")
    #         plt.plot(peaks, data[peaks], "x")
    #         # plt.plot(peaks, y3, label="3-curving")
    #         # plt.plot(peaks, y_curve, label="curve fitting")
    #         plt.legend()
    #         plt.show()

    def smoothing(self, x, data):
        # Savitzky–Golay smothing
        from scipy.signal import savgol_filter
        print(x.shape)
        print(data.shape)
        data = savgol_filter(np.array(data), 51, 3, mode="wrap")
        return data

    def get_envelopes(self):
        """
        Get envelopes by hilbert transformer
        """
        from scipy import signal
        from scipy.misc import electrocardiogram
        from scipy.signal import find_peaks
        x = electrocardiogram()[2000:4000]

        peaks, _ = find_peaks(x, distance=10)

    def get_curve(self, x, data):
        import scipy.interpolate as interp
        # 三次样条插值拟合 曲线。
        x = np.linspace(x.min(), x.max(), len(data))
        interp3 = interp.splrep(x, data, k=3)
        y3 = interp.splev(x, interp3)
        # 曲线拟合
        model = make_pipeline(PolynomialFeatures(8),
                              Ridge(alpha=0.1, fit_intercept=True))

        x_ploy = x.reshape(-1, 1)
        # x_ploy = x.reshape(-1, 1)
        model.fit(x_ploy, data)
        pred_data = model.predict(x_ploy)
        plt.plot(data, label="IOU")
        plt.plot(pred_data, label="pred_data")
        # plt.plot(data, label="IOU")
        plt.show()
        return x, y3, pred_data

    def plot_logistic_dis(self):
        # u = 0
        r = 180
        b = 0.06
        x = np.linspace(0, 1920, 214)
        # y = -np.exp(-(x - u) / r) / (r * (1 + np.exp(-(x - u) / r)) ** 2) + b
        # 归一化
        u = x.mean()
        print(u)
        sigma = 150
        y = np.abs(1.0 / (np.sqrt(2 * np.pi) * sigma) * (r * np.exp(-(x - u) ** 2 / (2 * sigma ** 2))) - 1) - b
        plt.plot(x, y)
        # plt.ylim(y.min(), y.max())
        plt.show()
        return x, y

    def curve_fitting(self, x, data):
        def func(x, u, r, b):
            return -np.exp(-(x - u) / r) / (r * (1 + np.exp(-(x - u) / r)) ** 2) + b

        from scipy.optimize import curve_fit

        u, r, b = curve_fit(func, x, data)[0]
        y_curve = -np.exp(-(x - u) / r) / (r * (1 + np.exp(-(x - u) / r)) ** 2) + b
        plt.plot(x, y_curve, label='curve_fit')
        plt.show()
        return y_curve


if __name__ == "__main__":
    cf = CurveFitting('.\iou_data\Truck_data.txt')
    peaks, x, data = cf.get_down_peaks(save_plot=True)
    # plt.plot(x[data > 0.5], data[data > 0.5])
    # plt.show()
    # cf.get_curve(x[data > 0.5], data[data > 0.5])
    # cf.plot_logistic_dis()
