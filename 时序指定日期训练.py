# 导入所需库
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os

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

data = load_and_filter_data(file_path,1960,2010)
Xtrain = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance","GM","GQ","GX","GD"]]
Ytrain = data['PRCP']

data = load_and_filter_data(file_path,2011,2018)
Xtest = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance","GM","GQ","GX","GD"]]
Ytest = data['PRCP']

#pd读取csv文件
data = pd.read_csv("F:\研究数据\台风风雨训练数据集\Rainingtestdatebase_1960-2018.csv", delimiter=',', index_col=["ID", "Time"], low_memory=False)
# 删除缺测值
data.dropna(axis=0, inplace=True)
X = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]  # 输入CSV文件的自变量X
Y = data['PRCP']  # 输入因变量(预报值) 可多选


# 创建随机森林回归模型
model = RandomForestRegressor(n_estimators=50, max_depth=40, oob_score=True, random_state=0, n_jobs=-1)
model.fit(Xtrain,Ytrain)

#准备评估模型
Ytest_p = model.predict(Xtest)
Ytrain_p = model.predict(Xtrain)
train=pd.DataFrame(Ytrain)
train['pred']=Ytrain_p
test=pd.DataFrame(Ytest)
test['pred']=Ytest_p
r_test = np.array(test.corr())[0,1]
r_train = np.array(train.corr())[0,1]

#模型评估参数计算
#相关系数（R）、偏差（Bias）、平均绝对误差（MAE）、均方根误差（RMSE）、观测值的标准偏差（SD0）、模拟值的标准偏差（SDP）以及模拟值与观测值的吻合指数IA
#公式详见formula.docx
train_mse = metrics.mean_squared_error(Ytrain, Ytrain_p)
train_rmse = np.sqrt(train_mse)
test_mse = metrics.mean_squared_error(Ytest, Ytest_p)
test_rmse = np.sqrt(test_mse)
bias_test = np.sum(Ytest_p-Ytest)/Ytest.size
bias_train = np.sum(Ytrain_p-Ytrain)/Ytrain.size
train_mae = metrics.mean_absolute_error(Ytrain, Ytrain_p)
test_mae = metrics.mean_absolute_error(Ytest, Ytest_p)
train_r = np.corrcoef(Ytrain, Ytrain_p)[0,1]
test_r = np.corrcoef(Ytest, Ytest_p)[0,1]
tmp1 = np.sqrt((Ytrain-Ytrain.mean())**2)
train_sdo = tmp1.sum()/(len(tmp1)-1)
tmp2 = np.sqrt((Ytest_p-Ytest_p.mean())**2)
test_sdp = tmp2.sum()/(len(tmp2)-1)
tmp3 = np.sqrt((Ytrain_p-Ytrain_p.mean())**2)
train_sdp = tmp3.sum()/(len(tmp3)-1)
tmp4 = np.sqrt((Ytest-Ytest.mean())**2)
test_sdo = tmp4.sum()/(len(tmp4)-1)
tmp = Ytrain-Ytrain_p
tmp5 = ((abs(Ytrain_p-Ytrain.mean())+abs(Ytrain-Ytrain.mean()))**2).sum()
tmp6 = (tmp**2).sum()
train_ia = 1-tmp6/tmp5
tmp = Ytest-Ytest_p
tmp5 = ((abs(Ytest_p-Ytest.mean())+abs(Ytest-Ytest.mean()))**2).sum()
tmp6 = (tmp**2).sum()
test_ia = 1-tmp6/tmp5

print("训练集：")
print("mae={:.2f}\nrmse={:.2f}\nbias={:.2f},\nr={:.2f}\nsdo={:.2f}\nsdp={:.2f}\nia={:.2f}".format(train_mae,train_rmse,bias_train,train_r,train_sdo,train_sdp,train_ia))
print("测试集：")
print("mae={:.2f}\nrmse={:.2f}\nbias={:.2f},\nr={:.2f}\nsdo={:.2f}\nsdp={:.2f}\nia={:.2f}".format(test_mae,test_rmse,bias_test,test_r,test_sdo,test_sdp,test_ia))

metrics_data = {
    "Dataset": ["Train", "Test"],
    "MAE": [round(train_mae, 2), round(test_mae, 2)],
    "RMSE": [round(train_rmse, 2), round(test_rmse, 2)],
    "Bias": [round(bias_train, 2), round(bias_test, 2)],
    "R": [round(train_r, 2), round(test_r, 2)],
    "SDO": [round(train_sdo, 2), round(test_sdo, 2)],
    "SDP": [round(train_sdp, 2), round(test_sdp, 2)],
    "IA": [round(train_ia, 2), round(test_ia, 2)]
}

metrics_df = pd.DataFrame(metrics_data)
save_path = os.path.join(current_path, "Model Evaluation.xlsx")
metrics_df.to_excel(save_path, index=False)

#绘评估结果图
x=np.linspace(0,300,300)
y=x
norm = matplotlib.colors.Normalize(vmax=-4) #vmin=-12, vmax=-5 vmin是人为设置待评估数据的最小值，vmax是最大值 数据值超出这个范围的部分将被映射为边界颜色

fig = plt.figure(figsize=(14,5)) #设置图像 单位英寸
gs=gridspec.GridSpec(8,27) #分割图像小单位格 把上述版面分割为8x27块 8行x27列

ax2=fig.add_subplot(gs[0:6,0:7])
ax=fig.add_subplot(gs[0:6,9:16])
ax3=fig.add_subplot(gs[0:6,18:25])
ax4=fig.add_subplot(gs[7,1:15])

# 上面生成了四个子图，ax是测试集的图（中间），ax2是测训练集的图（左边），ax3是特征重要性图（右边），ax4则是色条的位置（下面）
# ax.set_xticks(np.arange(0,350,50))
# ax.set_yticks(np.arange(0,350,50))
# np.ma掩码模块

# 获取二维直方分布
ax.tick_params(direction='inout')
nbins = 200
H_test, xe_test, ye_test = np.histogram2d(Ytest, Ytest_p, bins=nbins,density=True)
H_test = np.rot90(H_test)
H_test = np.flipud(H_test)
# 把0值掩，不画
Hmasked_test = np.ma.masked_where(H_test==0,H_test) 
Hmasked_test = np.ma.log(Hmasked_test)
# 开始绘图
c = ax.pcolormesh(xe_test, ye_test, Hmasked_test, cmap='jet',norm=norm)
ax.plot(y,':',c='green')
ax.set_xlabel('Concentration from Observations (mm)')
ax.set_xlim(0,250)
ax.set_ylim(0,250)
ax.set_ylabel('Concentration from Predictions (mm)')
ax.set_title('(b)', fontsize=18, loc='left')
#ax.text(20, 280, "$R$={:.3f}".format(float(r_test)),fontsize=12,verticalalignment="top",horizontalalignment="right")

ax2.tick_params(direction='inout') 
nbins = 200
H_train, xe_train, ye_train = np.histogram2d(Ytrain, Ytrain_p, bins=nbins,density=True)
H_train = np.rot90(H_train)
H_train = np.flipud(H_train)
Hmasked_train = np.ma.masked_where(H_train==0,H_train) 
Hmasked_train = np.ma.log(Hmasked_train)
ax2.pcolormesh(xe_train, ye_train, Hmasked_train, cmap='jet',norm=norm)
ax2.plot(y,':',c='black')
ax2.set_xlabel('Concentration from Observations (mm)')
ax2.set_xlim(0,250)
ax2.set_ylim(0,250)
ax2.set_ylabel('Concentration from Predictions (mm)')
ax2.set_title('(a)', fontsize=18, loc='left')

# 绘制特征重要性图
feature_importances = model.feature_importances_
features = X.columns
features_df = pd.DataFrame({'Features':features,'Importance':feature_importances})
features_df.sort_values('Importance',inplace=True,ascending=True)
ax3.barh(features_df.Features,features_df.Importance)
ax3.set_title('(c)', fontsize=18, loc='left')
# ax3.set_title('Feature Importances of $\mathrm{PREC}$')

cbar = plt.colorbar(c,ticks=np.arange(-10,-4,1),drawedges=False,cax=ax4,orientation='horizontal',fraction=0.07)
cbar.set_label('Frequency in log scale(%)')

print("最小频率值:", np.min(H_test))
print("最大频率值:", np.max(H_test))

save_path = os.path.join(current_path, "Figure test.jpg")
plt.savefig(save_path, dpi=1080, facecolor='white', bbox_inches='tight') # 保存图片