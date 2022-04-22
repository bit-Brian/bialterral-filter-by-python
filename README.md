# bialterral-filter-by-python
运用python实现的双边滤波器。此函数的缺点是比opencv的bilateralFilter函数运行速度慢，后续会对其进行修改，提升处理速度。本函数的优点是比opencv.bilateralFilter()函数的滤波效果好。本文参考了DuJunda（https://github.com/DuJunda/BilateralFilter/blob/master/BilateralFilter.py）的代码，在此表示感谢。此代码在其基础上实现了单通道和三通道图像的双边滤波，（DuJunda的代码不完善之处是不能处理单通道例如灰度图像）。
