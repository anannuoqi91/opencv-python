import cv2
import numpy as np
import Cameo.utils as utils


def recolorRC(src, dst):
    """Simulate conversion from BGR to RC (red, cyan).
    The source and destination images must both be in BGR format.
    Blues and greens are replaced with cyans. The effect is similar
    to Technicolor Process 2 (used in early color movies) and CGA
    Palette 3 (used in early color PCs).
    Pseudocode:
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).
    The source and destination images must both be in BGR format.
    Blues are desaturated. The effect is similar to Technicolor
    Process 1 (used in early color movies).
    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).
    The source and destination images must both be in BGR format.
    Yellows are desaturated. The effect is similar to CGA Palette 1
    (used in early color PCs).
    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


def blend(foregroundSrc, backgroundSrc, dst, alphaMask):
    # Calculate the normalized alpha mask.
    maxAlpha = np.iinfo(alphaMask.dtype).max
    normalizedAlphaMask = (1.0 / maxAlpha) * alphaMask

    # Calculate the normalized inverse alpha mask.
    normalizedInverseAlphaMask = \
        np.ones_like(normalizedAlphaMask)
    normalizedInverseAlphaMask[:] = \
        normalizedInverseAlphaMask - normalizedAlphaMask

    # Split the channels from the sources.
    foregroundChannels = cv2.split(foregroundSrc)
    backgroundChannels = cv2.split(backgroundSrc)

    # Blend each channel.
    numChannels = len(foregroundChannels)
    i = 0
    while i < numChannels:
        backgroundChannels[i][:] = \
            normalizedAlphaMask * foregroundChannels[i] + \
            normalizedInverseAlphaMask * backgroundChannels[i]
        i += 1

    # Merge the blended channels into the destination.
    cv2.merge(backgroundChannels, dst)


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """
    边缘检测：先对图像进行模糊处理（避免将噪声错误地识别为边缘），再转为灰度彩色图像，再使用laplacian检测边缘。
    对于典型的网络摄像头，blurKsize的值为7，edgeKsize的值为5，可能会产生最令人满意的效果。
    :param src: 待处理图像
    :param dst: 图像深度
    :param blurKsize: 中值滤波ksize
    :param edgeKsize: Laplacian算子ksize
    :return:
    """
    if blurKsize >= 3:  # ksize是基整数，大于等于3
        blurredSrc = cv2.medianBlur(src, blurKsize)  # 中值滤波
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 拉普朗斯算法边缘搜索：
    # 待处理图像（灰度）、
    # 通道深度：cv2.CV_8U表示每个通道8位）负值（例如，此例中为-1）表示目标图像和源图像有相同的深度、
    # 返回图像、
    # 核大小
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)  # 归一化
    # 分通道处理
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VFuncFilter(object):
    """
    用函数（vFunc）实例化的类，然后可以使用apply将其应用于图像。该函数适用于灰度图像的V（值）通道或彩色图像的所有通道。
    """
    def __init__(self, vFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView = np.ravel(src)
        dstFlatView = np.ravel(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView,
                               dstFlatView)


class VCurveFilter(VFuncFilter):
    """
    VFuncFilter的子类， 使用一组控制点（vPoints）实例化，这些控制点在内部用于创建曲线函数
    """

    def __init__(self, vPoints, dtype=np.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints),
                             dtype)


class BGRFuncFilter(object):
    """
    BGR图像使用，对应VFuncFilter（灰度）
    用最多四个函数实例化的类，然后可以使用apply将其应用于 BGR 图像。
    这些函数之一适用于所有通道（vFunc），而其他三个函数（bFunc，gFunc，rFunc）均适用于单个通道。
    首先应用整体函数，然后再应用每通道函数。
    """
    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """
        BGR影像分通道处理
        """
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """
    BGR图像使用
    BGRFuncFilter的子类，使用四组控制点实例化，这些控制点在内部用于创建曲线函数。
    对应VCurveFilter（灰度）
    """

    def __init__(self, vPoints=None, bPoints=None,
                 gPoints=None, rPoints=None, dtype=np.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """
    滤波器，模拟交叉处理
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 20), (255, 235)],
            gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
            rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
            dtype=dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    """
    Portra滤波器，模拟柯达胶片
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype=dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    """
    Provia滤波器，模拟富士胶片
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
            gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
            rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
            dtype=dtype)


class BGRVelviaCurveFilter(BGRCurveFilter):
    """
    Velvia滤波器，模拟富士维尔威亚胶片
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
            bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
            dtype=dtype)


class VConvolutionFilter(object):
    """
    一般的卷积滤波器
    """
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """
        感兴趣区域——中心像素点 * 权值 + 周围像素 * 权值
        """
        cv2.filter2D(src, -1, self._kernel, dst)


class BlurFilter(VConvolutionFilter):
    """
    滤波器
    对于模糊效果，权值的总和是1，而且整个邻域像素的权值都应该是正的
    """

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class SharpenFilter(VConvolutionFilter):
    """
    锐化滤波器
    """
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])  # 权值总和是1，保持图像整体亮度不变
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """
    滤波器
    边缘检测核，权值之和是0，使边缘变为白色，非边缘变为黑色
    """

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """
    滤波器
    使一边模糊（正权值），另一边锐化（负权值），会产生脊状或者浮雕的效果
    """
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
