import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb

# 1. 读取数据
train_path = 'data/used_car_train_20200313.csv'
test_path = 'data/used_car_testB_20200421.csv'
train_df = pd.read_csv(train_path, delim_whitespace=True)
test_df = pd.read_csv(test_path, delim_whitespace=True)

# 2. 特征选择（去除无关列）
features = [col for col in train_df.columns if col not in ['SaleID', 'price']]

# 3. 简单处理类别特征
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        le = LabelEncoder()
        all_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(all_data)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

# 4. 缺失值填充
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

# 5. 划分训练集和验证集
X = train_df[features]
y = train_df['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用sk-lean中的超参数优化
# 初始化模型（不传参数）
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 设置参数搜索空间
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 使用 GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # 目标是最小化MSE，取负值是sklearn规范
    cv=3,  # 三折交叉验证
    verbose=1,
    n_jobs=-1  # 多线程加速
)

# 开始搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最优模型预测
best_model = grid_search.best_estimator_


# 7. 验证集评估
y_pred_val = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)
print(f'验证集RMSE: {rmse:.2f}')
print(f'验证集MAE: {mae:.2f}')

# 8. 测试集预测
X_test = test_df[features]
test_pred = best_model.predict(X_test)

# 9. 生成提交文件
submit = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': test_pred})
submit['price'] = submit['price'].round(2)
submit.to_csv('xgboost_regression_submit.csv', index=False)
print('预测结果已保存到 xgboost_regression_submit.csv') 