import pandas as pd  # 数据分析
import matplotlib.pyplot as plt

# 若出现中文字符显示乱码，则添加以下两行，中文字符的输入格式为：“u"内容"”
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_train = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/train.csv")

plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u"获救情况 (1为获救)")  # 设定标题名称
plt.ylabel(u"人数")  # 设定纵坐标名称

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")
plt.ylabel(u"年龄")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.title(u"各等级的乘客年龄分布")
plt.xlabel(u"年龄")  # 设定横坐标名称
plt.ylabel(u"密度")  # 设定纵坐标名称
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # 设定图例

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

plt.show()

# 数据初步分析：
# ①891人中，得救人数300多，不到半数
# ②3等舱乘客很多，超过半数的乘客属于3等舱
# ③遇难和获救的人年龄跨度很广
# ④3个不同的舱年龄总体趋势一致，2、3等舱乘客20多岁的人最多，1等舱40岁左右的人最多
# ⑤登船港口人数按照S、C、Q递减，且S人数远多于另外2个港口

# 初步想法：
# ①不同舱位/乘客等级可能和财富、地位有关系，最后获救概率可能不一样
# ②年龄对获救概率有影响，小孩和女士先走
# ③和登船港口有关系，不同登船港口的人，地位不同
