# N维数组类型

import numpy as np

a = np.array([1, 2, 3])
print(a)
print()

a = np.array([[1, 2], [3, 4]])  # 2行2列
print(a)
print()

a = np.array([1, 2, 3, 4, 5], ndmin=2)
print(a)
print()

a = np.array([1, 2, 3], dtype=complex)
print(a)
print()

##############################

b = np.array([[1, 2, 3], [4, 5, 6]])  # 2行3列
print(b.shape)
print()

b = np.array([[1, 2, 3], [4, 5, 6]])
b.shape = (3, 2)  # 2行3列变为3行2列，shape改变原数组大小
print(b)
print()

b = np.array([[1, 2, 3], [4, 5, 6]])
c = b.reshape(3, 2)  # 2行3列变为3行2列，reshape不改变原数组大小
print(b)
print(c)
print()

##############################

c = np.arange(24)  # 生成等间隔数字的一维数组[0..23]（共24个元素）
print("数组维度:" + str(c.ndim))  # .ndim显示数组维度
d = c.reshape(2, 4, 3)  # 1维变为3维
print("数组维度:" + str(d.ndim))
print()

##############################

d = np.array([1, 2, 3, 4, 5], dtype=np.int8)  # dtype为int8(一个字节)
print("数组中每个元素的字节单位长度:" + str(d.itemsize))
d = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # dtype为float32(四个字节)
print("数组中每个元素的字节单位长度:" + str(d.itemsize))
print()

##############################

e = np.array([1, 2, 3, 4, 5])
print(e.flags)  # .flags返回ndarray对象的属性
