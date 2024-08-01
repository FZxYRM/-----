# 导入所需库
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os
from sklearn.model_selection import KFold
# 获取当前代码文件的绝对路径
current_code_path = os.path.abspath(__file__)
# 获取当前代码文件的目录路径
current_path = os.path.dirname(current_code_path)
file_path = "F:\研究数据\台风风雨训练数据集\Rainingtestdatebase_1960-2018.csv"

# 读取CSV文件并根据年份范围筛选数据 也可增加其他日期条件
def load_and_filter_data(file_path, start_year, end_year):
    # 读取CSV文件，注意设置encoding参数以避免中文字符编码问题
    data = pd.read_csv(file_path, delimiter=',', index_col=["ID"], low_memory=False, encoding='utf-8')
    # 将"Time"列转换为日期时间类型
    data['Time'] = pd.to_datetime(data['Time'], format='%Y%m%d%H', errors='coerce')
    data = data[(data['Time'].dt.year >= start_year) & (data['Time'].dt.year <= end_year)]
    data.dropna(axis=0, inplace=True)
    return data


#pd读取csv文件
data = pd.read_csv("F:\研究数据\台风风雨训练数据集\Rainingtestdatebase_1960-2018.csv", delimiter=',', index_col=["ID", "Time"], low_memory=False)
# 删除缺测值
data.dropna(axis=0, inplace=True)
X = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]  # 输入CSV文件的自变量X
Y = data['PRCP']  # 输入因变量(预报值) 可多选


# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# 初始化评估指标列表
mae_scores = []
rmse_scores = []
bias_scores = []
r_val_scores = []  # 用于存储验证集的R
r_train_scores = []  # 用于存储训练集的R
sdo_scores = []
sdp_scores = []
ia_scores = []

# 设置保存绘图的文件夹
save_folder = "K-Fold"

# 使用十折交叉验证进行模型评估
for k, (train_index, val_index) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]
    
    model = RandomForestRegressor(n_estimators=50, max_depth=40, oob_score=True, random_state=0, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    Y_pred_val = model.predict(X_val)
    Y_pred_train = model.predict(X_train)
    
    # 计算评估指标
    mae = metrics.mean_absolute_error(Y_val, Y_pred_val)
    rmse = np.sqrt(metrics.mean_squared_error(Y_val, Y_pred_val))
    bias = np.mean(Y_pred_val - Y_val)
    r_val = np.corrcoef(Y_val, Y_pred_val)[0, 1]
    r_train = np.corrcoef(Y_train, Y_pred_train)[0, 1]
    sdo = np.std(Y_val, ddof=1)
    sdp = np.std(Y_pred_val, ddof=1)
    ia = 1 - ((Y_pred_val - np.mean(Y_val)) ** 2).sum() / (((abs(Y_pred_val - np.mean(Y_val)) + abs(Y_val - np.mean(Y_val))) ** 2).sum())
    
    # 将评估指标添加到列表中
    mae_scores.append(round(mae))
    rmse_scores.append(round(rmse))
    bias_scores.append(round(bias))
    r_val_scores.append(round(r_val))
    r_train_scores.append(round(r_train))
    sdo_scores.append(round(sdo))
    sdp_scores.append(round(sdp))
    ia_scores.append(round(ia))

    # 绘制评估结果图
    x = np.linspace(0, 300, 300)
    y = x
    norm = matplotlib.colors.Normalize(vmin=-12, vmax=-5)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(8, 27)

    ax = fig.add_subplot(gs[0:6, 9:16])
    ax2 = fig.add_subplot(gs[0:6, 0:7])
    ax3 = fig.add_subplot(gs[0:6, 18:25])
    ax4 = fig.add_subplot(gs[7, 1:15])

    ax.tick_params(direction='in')
    ax.set_xticks(np.arange(0, 350, 50))
    ax.set_yticks(np.arange(0, 350, 50))

    nbins = 200
    H_test, xe_test, ye_test = np.histogram2d(Y_val, Y_pred_val, bins=nbins, density=True)
    H_test = np.rot90(H_test)
    H_test = np.flipud(H_test)
    Hmasked_test = np.ma.masked_where(H_test == 0, H_test)
    Hmasked_test = np.ma.log(Hmasked_test)
    
    c = ax.pcolormesh(xe_test, ye_test, Hmasked_test, cmap='jet', norm=norm)
    ax.plot(y, ':', c='green')
    ax.set_xlabel('Concentration from Observations (mm)')
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    ax.set_ylabel('Concentration from Predictions (mm)')
    ax.set_title(f'K-F{k} Testing Set of PRCP')
    ax.text(20, 280, "$R$={:.3f}".format(float(r_val)))


    ax2.tick_params(direction='in')
    ax2.set_xticks(np.arange(0, 350, 50))
    ax2.set_yticks(np.arange(0, 350, 50))

    nbins = 200
    H_train, xe_train, ye_train = np.histogram2d(Y_train, Y_pred_train, bins=nbins, density=True)
    H_train = np.rot90(H_train)
    H_train = np.flipud(H_train)
    Hmasked_train = np.ma.masked_where(H_train == 0, H_train)
    Hmasked_train = np.ma.log(Hmasked_train)

    ax2.pcolormesh(xe_train, ye_train, Hmasked_train, cmap='jet', norm=norm)

    ax2.plot(y, ':', c='black')
    ax2.set_xlabel('Concentration from Observations (mm)')
    ax2.set_xlim(0, 250)
    ax2.set_ylim(0, 250)
    ax2.set_ylabel('Concentration from Predictions (mm)')
    ax2.set_title(f'K-F{k} Training Set of PRCP')
    ax2.text(20, 280, "$R$={:.3f}".format(float(r_train)))

    cbar = plt.colorbar(c, ticks=np.arange(-12, -4, 1), drawedges=False, cax=ax4, orientation='horizontal', fraction=0.07)
    cbar.set_label('Frequency in log scale(%)')

    feature_importances = model.feature_importances_
    features = X.columns
    features_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
    features_df.sort_values('Importance', inplace=True, ascending=True)
    ax3.barh(features_df.Features, features_df.Importance)
    ax3.set_title('Feature Importances of $\mathrm{PREP}$')

    # 保存绘图
    save_path = f"{save_folder}/K-F{k}.png"
    plt.savefig(save_path, dpi=1080, facecolor='white')
    plt.close()


# 输出平均评估指标
avg_mae = np.round(np.mean(mae_scores),2)
avg_rmse = np.round(np.mean(rmse_scores),2)
avg_bias = np.round(np.mean(bias_scores),2)
avg_r_val = np.round(np.mean(r_val_scores),2)
avg_r_train = np.round(np.mean(r_train_scores),2)
avg_sdo = np.round(np.mean(sdo_scores),2)
avg_sdp = np.round(np.mean(sdp_scores),2)
avg_ia = np.round(np.mean(ia_scores),2)

results = []
for i in range(len(mae_scores)):
    results.append({
        'Fold': i + 1,
        'MAE': mae_scores[i],
        'RMSE': rmse_scores[i],
        'Bias': bias_scores[i],
        'R (Validation)': r_val_scores[i],
        'R (Training)': r_train_scores[i],
        'SD_o': sdo_scores[i],
        'SD_p': sdp_scores[i],
        'IA': ia_scores[i]
    })

# 创建平均结果数据框
avg_results = {
    'Fold': 'Average',
    'MAE': avg_mae,
    'RMSE': avg_rmse,
    'Bias': avg_bias,
    'R (Validation)': avg_r_val,
    'R (Training)': avg_r_train,
    'SD_o': avg_sdo,
    'SD_p': avg_sdp,
    'IA': avg_ia
}

results.append(avg_results)
results_df = pd.DataFrame(results)


# 保存到Excel文件
with pd.ExcelWriter('evaluation_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Fold Results', index=False)
    
    
# 打印指标
# for i in range(len(mae_scores)):
#     print(f'Fold {i+1} MAE: {mae_scores[i]:.2f}')
#     print(f'Fold {i+1} RMSE: {rmse_scores[i]:.2f}')
#     print(f'Fold {i+1} Bias: {bias_scores[i]:.2f}')
#     print(f'Fold {i+1} R (Validation): {r_val_scores[i]:.2f}')
#     print(f'Fold {i+1} R (Training): {r_train_scores[i]:.2f}')
#     print(f'Fold {i+1} SD_o: {sdo_scores[i]:.2f}')
#     print(f'Fold {i+1} SD_p: {sdp_scores[i]:.2f}')
#     print(f'Fold {i+1} IA: {ia_scores[i]:.2f}')
#     print()
# print(f'Average MAE: {avg_mae:.2f}')
# print(f'Average RMSE: {avg_rmse:.2f}')
# print(f'Average Bias: {avg_bias:.2f}')
# print(f'Average R (Validation): {avg_r_val:.2f}')
# print(f'Average R (Training): {avg_r_train:.2f}')
# print(f'Average SD_o: {avg_sdo:.2f}')
# print(f'Average SD_p: {avg_sdp:.2f}')
# print(f'Average IA: {avg_ia:.2f}')
