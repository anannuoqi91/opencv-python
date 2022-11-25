"""
date: 2022-11-24
describe:
1、修改截图、视频保存的文件名称
2、引用修改为绝对路径。在文件中引入了main入口，此时文件的__name__属性为‘__main__’而__package__属性为None，
如果在这样的文件中使用相对路径引入，解释器就找不到父级包的任何信息，会存在报错。
requirements:
opencv-contrib-python 4.6.0.66
python 3.7
"""


import cv2
from Cameo.capturemanager import CaptureManager
from Cameo.windowsmanager import WindowManager
import time


class Cameo(object):
    def __init__(self):
        # 创建窗口实例，窗口名称Cameo
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        # 视频管理类，有镜像功能，VideoCapture(0)多个设备时通过数字标定设备
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """
        主程序
        """
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                pass
            self._captureManager.exitFrame()  # 录屏、截屏
            self._windowManager.processEvents()  # 键盘控制

    def onKeypress(self, keycode):
        """
        键盘的回调函数
        space  -> 图像截图
        tab    -> 开始/结束视频录制
        escape -> 退出
        """
        if keycode == 32:  # space
            suffix = time.strftime("%Y%m%d-%H%M%S")
            self._captureManager.writeImage('screenshot-{}.png'.format(suffix))  # 图像名称
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                suffix = time.strftime("%Y%m%d-%H%M%S")
                self._captureManager.startWritingVideo(
                    'screencast-{}.avi'.format(suffix))
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
