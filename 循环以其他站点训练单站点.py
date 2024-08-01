# 导入所需库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前代码文件的绝对路径
current_code_path = os.path.abspath(__file__)
# 获取当前代码文件的目录路径
current_path = os.path.dirname(current_code_path)

# pd读取csv文件
data = pd.read_csv("test.csv", delimiter=',', index_col=["ID", "Time"], low_memory=False)
# 删除缺测值
data.dropna(axis=0, inplace=True)

X = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]  # 输入CSV文件的自变量X
Y = data['PRCP']  # 输入因变量(预报值)

# 获取所有站点的ID
stations = data.index.get_level_values("ID").unique()

# 初始化结果存储列表
results = []

# 轮流将每个站点的数据作为验证集，其他数据作为训练集
for station in stations:
    # 分割数据
    train_data = data[data.index.get_level_values("ID") != station]
    test_data = data[data.index.get_level_values("ID") == station]

    X_train = train_data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]
    Y_train = train_data['PRCP']
    X_test = test_data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]
    Y_test = test_data['PRCP']

    # 训练模型
    model = RandomForestRegressor(n_estimators=200, max_depth=50, oob_score=True, random_state=0, n_jobs=-1)
    model.fit(X_train, Y_train)

    # 预测
    Y_pred = model.predict(X_test)

    # 计算评估指标
    r = np.corrcoef(Y_test, Y_pred)[0, 1]
    bias = np.mean(Y_pred - Y_test)
    mase = np.mean(np.abs(Y_pred - Y_test)) / np.mean(np.abs(Y_test - np.mean(Y_test)))
    percentage_error = np.mean(np.abs((Y_pred - Y_test) / Y_test)) * 100

    # 保存结果
    results.append({
        "Station_ID": station,
        "R": r,
        "Bias": bias,
        "MASE": mase,
        "Percentage_Error": percentage_error
    })

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 保存结果到Excel文件
results_df.to_excel(os.path.join(current_path, "Evaluation_results.xlsx"), index=False)