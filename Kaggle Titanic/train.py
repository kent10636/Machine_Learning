import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame

data_train = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/train.csv")

# print(data_train)  # 打印整个训练集
# data_train.info()  # 打印dataframe简易信息（不同属性的行数，是否缺失，数据类型等）
# print(data_train.describe())  # 打印dataframe具体数值信息（不同属性的行数，均值等）
