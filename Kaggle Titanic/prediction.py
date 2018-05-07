import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import pickle

data_test = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/test.csv")

# 载入预处理后的train_data和test_data
df = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Processed_Data/processed_train_data.csv")
df_test = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Processed_Data/processed_test_data.csv")

# 载入训练模型
with open('E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Model/train_model.pkl', 'rb') as file:
    clf = pickle.load(file)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Prediction/logistic_regression_predictions.csv",
              index=False)
