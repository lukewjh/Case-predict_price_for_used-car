import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import xgboost as xgb

# 1. 读取特征工程后的数据
train_df = pd.read_parquet('train_fe.parquet')
test_df = pd.read_parquet('test_fe.parquet')

# 2. 特征选择（去除无关列）
features = [col for col in train_df.columns if col not in ['price']]
if 'SaleID' in features:
    print(features.colums)
    features.remove('SaleID')

# 3. 划分训练集和验证集
X = train_df[features]
y = train_df['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 使用sk-learn中的超参数优化
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    random_state=42,
    verbosity=2
)
fit_params = {
    "eval_set": [(X_val, y_val)],
    "eval_metric": "mae",
    "verbose": True  # 每一轮都打印
}
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    # scoring='neg_mean_squared_error',
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print('Best Parameters:', grid_search.best_params_)
best_model = grid_search.best_estimator_

# 5. 验证集评估
y_pred_val = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)
print(f'验证集RMSE: {rmse:.2f}')
print(f'验证集MAE: {mae:.2f}')

# 6. 测试集预测
X_test = test_df[features]
test_pred = best_model.predict(X_test)

# 7. 生成提交文件
submit = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': test_pred})
submit['price'] = submit['price'].round(2)
submit.to_csv('xgboost_regression_submit.csv', index=False)
print('预测结果已保存到 xgboost_regression_submit.csv') 