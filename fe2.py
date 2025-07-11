import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# ========== 1. 原始数据读取 ==========
train_path = 'data/used_car_train_20200313.csv'
test_path = 'data/used_car_testB_20200421.csv'
train_df = pd.read_csv(train_path, delim_whitespace=True)
test_df = pd.read_csv(test_path, delim_whitespace=True)

n_train = train_df.shape[0]
n_test = test_df.shape[0]
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ========== 2. 缺失值处理 ==========
all_data['notRepairedDamage_missing'] = all_data['notRepairedDamage'].eq('-').astype(int)
all_data['notRepairedDamage'] = all_data['notRepairedDamage'].replace('-', 'unknown')
# all_data = all_data.fillna(-1)
# 1. 先对所有object类型字段做LabelEncoder编码
# for col in all_data.columns:
#     if all_data[col].dtype == 'object':
#         le = LabelEncoder()
#         all_data[col] = le.fit_transform(all_data[col].astype(str))

# 2. 数值型字段用中位数填充，非数值型（已经编码为数字）用-1填充
for col in all_data.columns:
    if all_data[col].dtype in [np.float64, np.int64]:
        median = all_data[col].median()
        all_data[col] = all_data[col].fillna(median)
    else:
        all_data[col] = all_data[col].fillna(-1)

# ========== 3. 日期特征 ==========
all_data['regDate'] = pd.to_datetime(all_data['regDate'], format='%Y%m%d', errors='coerce')
all_data['creatDate'] = pd.to_datetime(all_data['creatDate'], format='%Y%m%d', errors='coerce')
all_data['regYear'] = all_data['regDate'].dt.year
all_data['regMonth'] = all_data['regDate'].dt.month
all_data['regDay'] = all_data['regDate'].dt.day
all_data['creatYear'] = all_data['creatDate'].dt.year
all_data['creatMonth'] = all_data['creatDate'].dt.month
all_data['carAge'] = ((all_data['creatDate'] - all_data['regDate']).dt.days / 365).clip(lower=0)
all_data = all_data.drop(['regDate', 'creatDate'], axis=1)

# ========== 4. 数值处理 ==========
all_data['power'] = pd.to_numeric(all_data['power'], errors='coerce').fillna(-1)
all_data['kilometer'] = pd.to_numeric(all_data['kilometer'], errors='coerce').fillna(-1)
all_data['power'] = np.clip(all_data['power'], 0, 600)
all_data['log_power'] = np.log1p(all_data['power'])
all_data['km_per_year'] = all_data['kilometer'] / (all_data['carAge'] + 0.1)

# ========== 5. 均值编码 ==========
train_df['brand'] = train_df['brand'].astype(str)
train_df['model'] = train_df['model'].astype(str)
all_data['brand'] = all_data['brand'].astype(str)
all_data['model'] = all_data['model'].astype(str)

brand_mean_price = train_df.groupby('brand')['price'].mean()
model_mean_price = train_df.groupby('model')['price'].mean()
all_data['brand_avg_price'] = all_data['brand'].map(brand_mean_price).fillna(brand_mean_price.mean())
all_data['model_avg_price'] = all_data['model'].map(model_mean_price).fillna(model_mean_price.mean())

# ========== 6. 类别编码 ==========
for col in all_data.columns:
    if all_data[col].dtype == 'object':
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

# ========== 7. 高级特征工程增强 ==========
# 保存 SaleID 和 name 字段备用
saleid = all_data['SaleID'] if 'SaleID' in all_data.columns else None
name = all_data['name'] if 'name' in all_data.columns else None

# 品牌+车型组合
all_data['brand_model'] = all_data['brand'].astype(str) + '_' + all_data['model'].astype(str)
brand_model_price_map = train_df.copy()
brand_model_price_map['brand_model'] = train_df['brand'].astype(str) + '_' + train_df['model'].astype(str)
brand_model_mean_price = brand_model_price_map.groupby('brand_model')['price'].mean()
all_data['brand_model_avg_price'] = all_data['brand_model'].map(brand_model_mean_price).fillna(brand_model_mean_price.mean())

# power 分桶
all_data['power_bin'] = pd.qcut(all_data['power'], 5, labels=False, duplicates='drop')
power_bin_price = train_df.copy()

# 强制转换为数值型，无法转换的变为NaN
train_df['power'] = pd.to_numeric(train_df['power'], errors='coerce')
# 用中位数填充NaN
train_df['power'].fillna(train_df['power'].median(), inplace=True)

power_bin_price['power_bin'] = pd.qcut(train_df['power'], 5, labels=False, duplicates='drop')
power_bin_mean_price = power_bin_price.groupby('power_bin')['price'].mean()
all_data['power_bin_avg_price'] = all_data['power_bin'].map(power_bin_mean_price).fillna(power_bin_mean_price.mean())

# 每千瓦年行驶里程
all_data['km_per_kw_year'] = all_data['kilometer'] / (all_data['power'] + 0.1) / (all_data['carAge'] + 0.1)

# KFold target encoding：brand
def kfold_target_encoding(df, col, target, n_splits=5):
    df[col + '_te'] = 0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        df.loc[df.index[val_idx], col + '_te'] = df.iloc[val_idx][col].map(means)
    df[col + '_te'] = df[col + '_te'].fillna(df[target].mean())
    return df[col + '_te']

train_te = all_data.iloc[:n_train].copy()
train_te['brand_te'] = kfold_target_encoding(train_te, 'brand', 'price')
brand_te_map = train_te[['brand', 'brand_te']].drop_duplicates().set_index('brand')['brand_te'].to_dict()
all_data['brand_te'] = all_data['brand'].map(brand_te_map).fillna(train_te['brand_te'].mean())

# 删除临时特征
all_data.drop(['brand_model'], axis=1, inplace=True)

# ========== 8. 删除不参与训练的字段 ==========
drop_cols = []
if 'SaleID' in all_data.columns:
    drop_cols.append('SaleID')
if 'name' in all_data.columns:
    drop_cols.append('name')
all_data = all_data.drop(columns=drop_cols)

# ========== 9. 拆分训练测试并恢复 SaleID / name 到测试集 ==========
train_fe = all_data.iloc[:n_train, :].copy()
test_fe = all_data.iloc[n_train:, :].copy()

if saleid is not None:
    print("开始将SaleID放回test集")
    test_fe['SaleID'] = saleid.iloc[n_train:].values
if name is not None:
    print("开始将name放回test集")
    test_fe['name'] = name.iloc[n_train:].values

# 可选：将 SaleID 和 name 放前面
cols = test_fe.columns.tolist()
for col in ['SaleID', 'name']:
    if col in cols:
        cols.remove(col)
        cols = [col] + cols
test_fe = test_fe[cols]

# ========== 10. 保存 ==========
if 'price' in test_fe.columns:
    print("dropping price col.....")
    test_fe = test_fe.drop(columns=['price'])

print(test_fe.columns)
train_fe.to_parquet('train_fe.parquet', index=False)
test_fe.to_parquet('test_fe.parquet', index=False)

print("✅ 增强型特征工程完成并保存。")

# ------------------------ 接下来执行你已有的训练流程 ------------------------

# 1. 加载增强特征后的数据
# train_df = pd.read_parquet('train_fe.parquet')
# test_df = pd.read_parquet('test_fe.parquet')

# # 2. 特征选择
# features = [col for col in train_df.columns if col not in ['price']]
# if 'SaleID' in features: features.remove('SaleID')
# if 'name' in features: features.remove('name')

# # 3. 数据划分
# X = train_df[features]
# y = train_df['price']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. XGBoost + GridSearchCV
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
# print('Best Parameters:', grid_search.best_params_)
# best_model = grid_search.best_estimator_

# # 5. 验证集评估
# y_pred_val = best_model.predict(X_val)
# rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
# mae = mean_absolute_error(y_val, y_pred_val)
# print(f'验证集RMSE: {rmse:.2f}')
# print(f'验证集MAE: {mae:.2f}')

# # 6. 测试集预测
# X_test = test_df[features]
# test_pred = best_model.predict(X_test)

# # 7. 生成提交文件
# submit = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': test_pred.round(2)})
# submit.to_csv('xgboost_regression_submit.csv', index=False)
# print('✅ 预测结果已保存为 xgboost_regression_submit.csv')
