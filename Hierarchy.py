import numpy as np
import lightgbm as lgb
import pandas as pd
from pyhts import Hierarchy
from pyhts import HFModel
from darts import TimeSeries
import warnings 
warnings.filterwarnings("ignore")
df = pd.read_csv('Hier.csv')
test = df[['Product_1','Product_2','Product_3']].transpose()
test.insert(0,'Batch',['ONE','ONE','ONE'])
test.insert(1,'Product',['A','B','C'])
test = test.reset_index(drop=True)
hierarchy = Hierarchy.new(test,structures=[('Batch','Product')])
Base_model = HFModel(hierarchy, base_forecasters=None, hf_method='comb', comb_method='ols')
#top层为0列 
all_levels = hierarchy.aggregate_ts(test.iloc[:,2:].transpose())
data_index = pd.to_datetime(df["Date"], format="%Y-%m-%d")
# 把时间序列当成索引
all_levels.index = data_index
all_levels
#NBEATS方法预测 
#############################   NBEATS预测只需要时间序列信息（目标变量）（pyhts处理后的序列）
from darts.models import NBEATSModel
series2 = TimeSeries.from_series(all_levels[0])
series2.plot()
train2, val2 = series2[:700], series2[700:]
model_nbeats = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=50,
    nr_epochs_val_period=1,
    batch_size=800,
    model_name="nbeats_run",
)
model_nbeats.fit(train2, val_series=val2, verbose=True)
pred_series = model_nbeats.historical_forecasts(
    series2,
    start=pd.Timestamp("20061202"),
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=True,
)
#底层LGB训练
#######################   LGB需要输入特征变量（原始数据集，pyhts处理前），与NBEATS使用的不是同一个序列
#设定损失函数
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * LOSS_MULTIPLIER)
    hess = np.where(residual < 0, 2, 2 * LOSS_MULTIPLIER)
    return grad, hess
#参数设置 
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric':'rmse',
        'n_jobs': -1,
        'seed': 42,
        'learning_rate': 0.2,
        'bagging_fraction': 0.85,
        'bagging_freq': 1, 
        'colsample_bytree': 0.85,
        'colsample_bynode': 0.85,
        'min_data_per_leaf': 25,
        'num_leaves': 200,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
         'verbose': -1
}
train_data1 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][0:700],df['Product_1'][0:700])
valid_data1 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][700:],df['Product_1'][700:])
train_data2 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][0:700],df['Product_2'][0:700])
valid_data2 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][700:],df['Product_2'][700:])
train_data3 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][0:700],df['Product_3'][0:700])
valid_data3 = lgb.Dataset(df[['d','Week','Weekend','Day','Month','Holiday']][700:],df['Product_3'][700:])
#寻找最优lamda
#x为序列数量 'd','Week','Weekend','Day','Month','Holiday'等特征作为函数输入
x = 3
arg = 10**10
lamda = np.linspace(1,1.5,40)
lamdabest = 0
for i in lamda:
    LOSS_MULTIPLIER = i
    #训练LGB模型
    for j in range(1,x+1):
        locals()['estimator' + str(j)] = lgb.train(lgb_params,
                      eval('train_data'+str(j)),
                      num_boost_round = 3600, 
                      early_stopping_rounds = 50, 
                      valid_sets = [eval('train_data'+str(j)), eval('valid_data'+str(j))],
                      verbose_eval = 100,
                      fobj = custom_asymmetric_train
                      )
        locals()['predict' + str(j)] = eval('estimator'+str(j)).predict(df[['d','Week','Weekend','Day','Month','Holiday']][700:])
    bottom_up = predict1+predict2+predict3
    arg_new = sum((pred_series.values().reshape(-1)-bottom_up)**2)
    if arg_new<arg:
        arg = arg_new
        lamdabest = i
    else:
        arg = arg
    print(i,arg,lamdabest)
print('最优lamda为：',lamdabest)
LOSS_MULTIPLIER_final = lamdabest
#训练模型
for j in range(1,x+1):
    locals()['estimator_final' + str(j)] = lgb.train(lgb_params,
                      eval('train_data'+str(j)),
                      num_boost_round = 3600, 
                      early_stopping_rounds = 50, 
                      valid_sets = [eval('train_data'+str(j)), eval('valid_data'+str(j))],
                      verbose_eval = 100,
                      fobj = custom_asymmetric_train
                      )
    locals()['predict_final' + str(j)] = eval('estimator_final'+str(j)).predict(df[['d','Week','Weekend','Day','Month','Holiday']][700:])
#迭代寻找lamda
bottom_up_final = predict_final1+predict_final2+predict_final3
#结果展示
result = {'Real':pd.Series(all_levels[0][700:],index=pd.to_datetime(df["Date"][700:], format="%Y-%m-%d")),'Prediction':pd.Series(bottom_up_final,index=pd.to_datetime(df["Date"][700:], format="%Y-%m-%d"))}
result = pd.DataFrame(result) 
series_result = TimeSeries.from_dataframe(result)
series_result.plot()