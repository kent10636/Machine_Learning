# 根据预处理后的train_data建模
import pandas as pd  # 数据分析

df = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Processed_Data/processed_train_data.csv")

# 把processed_train_data.csv中需要的feature字段取出来，转成numpy格式，使用scikit-learn的LogisticRegression建模
from sklearn import linear_model

# 用正则表达式取出需要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

# print(clf)
import pickle

with open('E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Model/train_model.pkl', 'wb')as file:
    pickle.dump(clf, file)
