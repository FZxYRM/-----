# 导入所需库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# 获取当前代码文件的绝对路径
current_code_path = os.path.abspath(__file__)
# 获取当前代码文件的目录路径
current_path = os.path.dirname(current_code_path)

#pd读取csv文件
data = pd.read_csv("F:\研究数据\台风风雨训练数据集\Rainingtestdatebase_1960-2018.csv", delimiter=',', index_col=["ID", "Time"], low_memory=False)

# 添加筛选条件
# data = data[data['Distance'] < 1000]

# 删除缺测值
data.dropna(axis=0, inplace=True)

X = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]  # 输入CSV文件的自变量X
Y = data['PRCP']  # 输入因变量(预报值) 可多选

#随机分割数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size=0.9, random_state=1)#随机分割 适合每行数据独立，如果是天气时序数据集应该顺序分割

#时序分割数据集
# train_size = int(len(X) * 0.9)
# Xtrain, Xtest = X[:train_size], X[train_size:]
# Ytrain, Ytest = Y[:train_size], Y[train_size:]

#建模
model = RandomForestRegressor(n_estimators=200, max_depth=50, oob_score=True, random_state=0, n_jobs=-1)
model.fit(Xtrain, Ytrain)

# 读取CSV文件
file_path = r"Validation_data.CSV"
data = pd.read_csv(file_path)
# 列名
clm = ["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance","GM","GQ","GX","GD"]
# 创建一个空列表，用于存储每行的预测值
y_pred_list = []
# 遍历每一行数据
for index, row in data.iterrows():
    df = pd.DataFrame([row[clm]], columns=clm)
    y_pred = model.predict(df)
    # 将预测值添加到原始数据的新列中，这里假设新增的列名为 'PRCP_PRED'
    data.at[index, 'PRCP_PRED'] = y_pred[0]
    # 将预测值添加到列表中
    y_pred_list.append(y_pred[0])
# 将预测结果写回CSV文件，这将添加一个新列 'PRCP_PRED'
data.to_csv(file_path, index=False)