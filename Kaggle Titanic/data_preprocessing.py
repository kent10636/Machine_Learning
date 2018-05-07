# train_data和test_data的数据预处理，以及逻辑回归建模
import pandas as pd

data_train = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/train.csv")
data_test = pd.read_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Data/test.csv")

##################################################
# 查看并分析原训练集
# print(data_train)  # 打印整个训练集
# data_train.info()  # 打印dataframe简易信息（不同属性的行数，是否缺失，数据类型等）
# print(data_train.describe())  # 打印dataframe具体数值信息（不同属性的行数，均值等）

##################################################
# 用scikit-learn中的RandomForest来拟合缺失的年龄数据
from sklearn.ensemble import RandomForestRegressor


# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
# print(data_train)

##################################################
# 特征因子化（即将属性的不同维度全部划分为一个单独属性列），把类目属性全都转成0、1的数值属性
# 以Sex为例，原本是一个属性维度，其取值可以是['male','female']，将其平展开为'Sex_male','Sex_female'两个属性
# 原取值为male的，其Sex_male取值为1，Sex_female取值为0；原取值为female的，其Sex_male取值为0，Sex_female取值为1
# 使用pandas的"get_dummies"，拼接在原"data_train"上
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(df)

##################################################
# 用scikit-learn的preprocessing模块对Age和Fare两个属性做scaling（缩放），将一些变化幅度较大的特征化到[-1,1]之内
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))  # .values.reshape(-1,1)表示将1D array变换为2D array
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

# print(df)  # 可以看到Age_scaled和Fare_scaled已经特征化为[-1,1]之间了
df.to_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Processed_Data/processed_train_data.csv", index=False)
# train_data预处理完成

##################################################
# 也要对test_data做同样的数据预处理

data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

# 用RandomForestRegressor模型填上丢失的年龄Age
tmp_age_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_age_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
data_test = set_Cabin_type(data_test)

# 特征因子化
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# 缩放Age和Fare
tmp_age_scale_param = scaler.fit(df_test['Age'].values.reshape(-1, 1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), tmp_age_scale_param)
tmp_fare_scale_param = scaler.fit(df_test['Fare'].values.reshape(-1, 1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), tmp_fare_scale_param)

# print(df_test)
df_test.to_csv("E:/PycharmProjects/Machine_Learning/Kaggle Titanic/Processed_Data/processed_test_data.csv", index=False)
# test_data预处理完成
