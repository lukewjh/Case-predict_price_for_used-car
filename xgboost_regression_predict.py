import pandas as pd
from sklearn.model_selection import train_test_split
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

# 6. 训练XGBoost回归模型
model = xgb.XGBRegressor(
    n_estimators=100,       # 树的数量
    max_depth=6,            # 树的最大深度
    learning_rate=0.1,      # 学习率
    subsample=0.8,          # 每棵树使用的样本比例
    colsample_bytree=0.8,   # 每棵树使用的特征比例
    gamma=0,                # 最小损失函数下降值（控制是否分裂）
    reg_alpha=0,            # L1 正则
    reg_lambda=1,           # L2 正则
    objective='reg:squarederror', # 回归目标函数（平方误差）
    random_state=42
)
model.fit(X_train, y_train)

# 7. 验证集评估
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)
print(f'验证集RMSE: {rmse:.2f}')
print(f'验证集MAE: {mae:.2f}')

# 8. 测试集预测
X_test = test_df[features]
test_pred = model.predict(X_test)

# 9. 生成提交文件
submit = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': test_pred})
submit['price'] = submit['price'].round(2)
submit.to_csv('xgboost_regression_submit.csv', index=False)
print('预测结果已保存到 xgboost_regression_submit.csv') 