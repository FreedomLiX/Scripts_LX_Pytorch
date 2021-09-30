import os
import cv2
from threading import Thread
from queue import Queue
import glob
import time
import numpy as np


class Video2Pic:
    # videos to pictures 利用 线程 threading 和 队列queue data structure
    def __init__(self, video_path, pic_path, rate):
        # 视频图像路径：‘./datasets/video/video.mp4’
        # 图像路径：'./datasets/picture'
        self.__video_path = video_path
        self.__pic_path = pic_path
        self.__rate = rate

        # 依视频文件名、创建文件夹，保存图片路径
        if not os.path.isfile(self.__video_path):
            raise Exception('ERROR: %s does not exist' % self.__video_path)
        if not os.path.isdir(self.__pic_path):
            raise Exception('ERROR: %s does not exist' % self.__pic_path)
        video_name = self.__video_path.split('/')[-1][:-4]
        self.picture_folder = os.path.join(self.__pic_path, video_name)
        # assert self.img_files, 'No images found'
        self.cap = None
        self.img = None
        # 线程队列
        # self.queue = Queue(maxsize=128)
        self.count = 1
        self.index = 1

    def start(self):
        # 开启线程进行视频读取，文件保存
        thread = Thread(target=self.update(), args=(), daemon=True)
        thread.start()
        return self

    def update(self):
        # 线程队列进行保存
        print("【INFO】: starting read {} video and save pictures".format(video))
        print(self.__video_path)
        self.cap = cv2.VideoCapture(self.__video_path)
        assert self.cap.isOpened(), 'Failed to open %s' % video
        # 图像宽、高、帧率 信息
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) % 100
        # 图片，文件保存路径
        if not os.path.exists(self.picture_folder):
            os.makedirs(self.picture_folder)

        # Read frame in thread
        while True:
            ret, self.img = self.cap.read()
            if not ret:
                self.cap.release()
                break
            self.count += 1
            if self.count == self.__rate:
                self.write(self.img)
                self.index += 1
                self.count = 1
            time.sleep(0.01)  # wait time

    # def read(self):
    #     return self.queue.get()
    #
    # def queue_size(self):
    #     return self.queue.qsize() > 0

    def write(self, img):
        print("【INFO】：video:{},to pictures".format(
            self.__video_path.split('/')[-1]))
        cv2.imencode('.jpg', img)[1].tofile(
            self.picture_folder + '/' + str(self.picture_folder.split('/')[-1])
            + '_' + str(self.index) + '.jpg')


# 多线程读取多路视频数据，并将多路视频‘stack’
class LoadMultiStreams:  # multiple videos
    def __init__(self, video_path, img_size=640):
        self.video_path = video_path
        self.mode = 'images'
        self.img_size = img_size
        video_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
        files = sorted(glob.glob(os.path.join(self.video_path, '*.*')))
        sources = [x for x in files if x.split('.')[-1].lower() in video_formats]

        # if os.path.isfile(sources):
        #     with open(sources, 'r') as f:
        #         sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        # else:
        #     sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([self.letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [self.letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


def multi_videos2pic(videos_path, pic_path):
    # 视频格式
    video_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
    files = sorted(glob.glob(os.path.join(videos_path, '*.*')))
    videos = [x for x in files if x.split('.')[-1].lower() in video_formats]
    for video in videos:
        cap = Video2Pic(video, pic, 75)
        cap.start()


if __name__ == '__main__':
    video = '/home/dixn/PycharmProjects/Utils/datasets/videos'
    pic = '../datasets/uuu'
    # multi_videos2pic(video, pic)
    dataset = LoadMultiStreams(video)
    for sources, img, img0, _ in dataset:
        print(sources)
        print(len(img0))
        print(img.shape)



