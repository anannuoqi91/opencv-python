"""
describe: 视频管理类，用来读取新的一帧图像，并将图像分派到一个或多个输出（图片文件、视频文件、窗口）中
"""


import cv2
import numpy as np
import time


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        """
        初始化实例
        :param capture: 摄像头设备
        :param previewWindowManager: WindowsManager的实例
        :param shouldMirrorPreview: bool变量，是否镜像操作
        """
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        # 私有变量，建议只在类及其子类中访问
        self._capture = capture
        self._channel = 0  # 通道，在这个项目中没有用到
        self._enteredFrame = False  # 是否抓取到一帧图片
        self._frame = None  # 当前帧图片
        self._imageFilename = None  # 写入的图像文件名，截图时提供，截图完成后重新赋值为None
        self._videoFilename = None  # 写入的视频文件名，截取视频时提供，录制结束后重新赋值为None
        self._videoEncoding = None  # 视频编码，录制视频时使用，录制结束后重新赋值为None
        self._videoWriter = None   # 视频写对象，录制视频时使用，录制结束后重新赋值为None

        self._startTime = None  # 结算fps的起始时间
        self._framesElapsed = 0  # 视频录制帧数
        self._fpsEstimate = None  # fps，每秒帧数

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        """
        为channel设定值
        :param value:
        :return:
        """
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        """
        _enteredFrame判断抓取图片是否成功，成功则把图片的数据导入到_frame中
        :return:
        """
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(
                self._frame, self.channel)
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """
        启动摄像头录制功能
        如果_capture不为空的话，抓取摄像头中的一帧
        """
        # 判断之前的窗口是否存在,若存在报错
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()  # 指向下一个帧
            # read是grab和retrieve的结合，grab是指向下一个帧，retrieve是解码并返回一个
            # 帧，而且retrieve比grab慢一些，所以当不需要当前的帧或画面时，可以使用grab跳过，与其使用read更省时间。
            # 因为有的时候缓冲区的画面是存在了延迟的。当不需要的时候可以多grab之后再read的话，就能比一直read更省时间。

    def exitFrame(self):
        """
        计算摄像头每秒钟的帧数，并且把摄像头捕获的帧放入窗口cameo中显示、截图保存、视频保存。
        :return:
        """
        # 检查是否有任何抓取的帧可检索，即判断enterFrame是否有抓取到一帧
        if self.frame is None:
            self._enteredFrame = False  # _enteredFrame为False时,为第一次抓取视屏信息,采用enterFrame中的grab()方式
            # 为True时为非第一次抓取，采用retrieve方式抓取，同时为frame赋值
            return
        # 计算每秒帧数
        if self._framesElapsed == 0:  # 初始值为0
            self._startTime = time.perf_counter()  # 首次记录开始时间，程序的执行时间
        else:
            timeElapsed = time.perf_counter() - self._startTime  # 非首次则计算当前与首次的时间间隔
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # 把捕获的帧的图像显示到屏幕窗口上面
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame)
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # 根据_imageFilename判定是否保存截图
        # 触发机制：判断是否按下了空格键，如果按下了，就把我们按下时候刚好_enterFrame捕获的那一帧数据保存到硬盘中。如果没有截图，就跳过
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None  # 截图完成，重新赋值为None

        # 保存视频
        self._writeVideoFrame()

        # #录像生成，将当前帧图像全部变为空
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """

        :param filename:
        :return:
        """
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        """
        开始录制
        :param filename:
        :param encoding:
        :return:
        """
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """
        停止录制
        :return:
        """
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        """
        触发机制Tab
        就把按下Tab时候刚好_enterFrame捕获的那一帧数据保存到硬盘中，
        然后每一次循环的时候都把_enterFrame保存起来，直到又按了一下TAB键就不再保存了。
        这样好多帧的图片会放到一个为avi格式的文件当中就是一个视频了。
        :return:
        """
        if not self.isWritingVideo:  # 由_videoFilename判定
            return

        if self._videoWriter is None:  # 如果视频保存方式无定义
            fps = self._capture.get(cv2.CAP_PROP_FPS)  # 获取fps
            if np.isnan(fps) or fps <= 0.0:
                # 如果未获取到fps
                if self._framesElapsed < 20:
                    # 如果现有帧数小于20,无法计算直接退出子程序
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)  # 创建videoWriter

        self._videoWriter.write(self._frame)
