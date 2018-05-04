import pandas as pd  # 数据分析
import matplotlib.pyplot as plt

# 若出现中文字符显示乱码，则添加以下两行，中文字符的输入格式为：“u"内容"”
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_train = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/train.csv")

# 各等级的获救情况：等级为1的乘客，获救概率高很多
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()

##################################################
# 各性别的获救情况：女性乘客获救概率远大于男性
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况(1为获救)")
plt.xlabel(u"获救情况")
plt.ylabel(u"人数")
plt.show()

##################################################
# 各乘客等级下各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.65)  # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况（1、2等乘客坐高级舱，3等乘客坐低级舱）")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                            label="female highclass",
                                                                                            color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                            label='female, low class',
                                                                                            color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                          label='male, high class',
                                                                                          color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                          label='male low class',
                                                                                          color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()

##################################################
# 各登船港口的获救情况
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登船港口乘客的获救情况")
plt.xlabel(u"登船港口")
plt.ylabel(u"人数")
plt.show()

##################################################
# 堂兄弟/姐妹、父母/孩子人数，对获救的影响：未发现明显关联
# g = data_train.groupby(['SibSp', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
# g = data_train.groupby(['Parch', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
# print()

##################################################
# 根据有无cabin信息查看的获救情况：有Cabin记录的获救概率稍高一些
# print(data_train.Cabin.value_counts())  # 统计每个客舱号的人数
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况(1为获救)")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
plt.show()
