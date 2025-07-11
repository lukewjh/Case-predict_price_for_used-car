import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from catboost import CatBoostRegressor, Pool

# 1. 读取特征工程后的数据
train_df = pd.read_parquet('train_fe.parquet')
test_df = pd.read_parquet('test_fe.parquet')

# 2. 特征选择（去除无关列）
features = [col for col in train_df.columns if col not in ['price']]
if 'SaleID' in features:
    features.remove('SaleID')

# 自动识别类别特征（CatBoost 可以直接用字符串型特征）
cat_features = [col for col in features if train_df[col].dtype == 'object']

# 3. 划分训练集和验证集
X = train_df[features]
y = train_df['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 使用sk-learn中的超参数优化
cat_model = CatBoostRegressor(
    loss_function='RMSE',
    random_state=42,
    verbose=100
)

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 500],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# 使用 Pool 提供类别特征索引
train_pool = Pool(X_train, y_train, cat_features=cat_features)
grid_search.fit(X_train, y_train, cat_features=cat_features)

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
submit.to_csv('catboost_regression_submit.csv', index=False)
print('预测结果已保存到 catboost_regression_submit.csv')
