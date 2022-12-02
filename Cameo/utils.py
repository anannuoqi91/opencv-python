"""
describe: 使用“曲线”过滤器弯曲颜色空间的通用的数学函数
"""


import numpy
import scipy.interpolate


def createLookupArray(func, length=256):
    """
    将为给定函数创建查找数组，查找值范围为 [0, length - 1]
    通常只处理 256 个可能的输入值（每个通道8位），并且可以廉价地预先计算并存储许多输出值。
    然后，我们的每通道每像素成本只是对缓存的输出值的查找。
    """
    if func is None:
        return None
    lookupArray = numpy.empty(length)  # 查找数组初始化
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray


def applyLookupArray(lookupArray, src, dst):
    """
    将查找数组（例如前一个函数的结果lookupArray）应用于另一个数组（例如图像src）
    """
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]


def createCurveFunc(points):
    """
    将控制点转换为函数
    插入任意控制点的曲线的函数
    """
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    if numPoints < 3:  # 线性插值
        kind = 'linear'
    elif numPoints < 4:  # 三个控制点，二次插值
        kind = 'quadratic'
    else:  # 至少4个控制点，三次差值
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)  # 接受两个数组（x和y坐标）并返回一个对点进行插值的函数


def createCompositeFunc(func0, func1):
    """
    将两个曲线函数合为一个曲线函数，仅限于采用单个参数的输入函数
    解决连续应用多个曲线的问题
    """
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))
