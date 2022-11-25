"""
describe: 窗口管理类，使应用程序代码能以面向对象的形式处理窗口的事件
"""


import cv2


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        """
        初始化实例
        :param windowName: 窗口名称
        :param keypressCallback: 实现按键控制功能
        """
        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False  # 控制是否循环提取摄像头信息

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        """
        创建窗口，用来在屏幕上现实摄像头内容
        :return:
        """
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True  # 创建成功

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)
