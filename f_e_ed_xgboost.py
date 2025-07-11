import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import xgboost as xgb

# 1. 读取特征工程后的数据
train_df = pd.read_parquet('train_fe.parquet')
test_df  = pd.read_parquet('test_fe.parquet')

# 2. 特征选择（去除无关列）
features = [c for c in train_df.columns if c not in ['price']]
if 'SaleID' in features:
    features.remove('SaleID')

# 3. 划分训练集和验证集
X = train_df[features]
y = train_df['price']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 直接实例化 XGBRegressor（手动设定超参数）
xgb_model = xgb.XGBRegressor(
    tree_method      = "hist",
    grow_policy      = "lossguide",
    max_depth        = 0,
    max_leaves       = 512,
    min_child_weight = 5,

    learning_rate    = 0.05,
    n_estimators     = 2000,

    subsample        = 0.8,
    colsample_bytree = 0.8,

    reg_lambda       = 2.0,
    reg_alpha        = 0.0,

    objective        = "reg:absoluteerror",
    eval_metric      = "mae",
    random_state     = 42,
    verbosity        = 2,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# 6. 验证集评估
y_pred_val = xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae  = mean_absolute_error(y_val, y_pred_val)
print(f'验证集 RMSE: {rmse:.2f}')
print(f'验证集 MAE : {mae:.2f}')

# 7. 测试集预测
X_test   = test_df[features]
test_pred = xgb_model.predict(X_test)

# 8. 生成提交文件
submit = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price' : np.round(test_pred, 2)
})
submit.to_csv('xgboost_regression_submit.csv', index=False)
print('预测结果已保存到 xgboost_regression_submit.csv')
