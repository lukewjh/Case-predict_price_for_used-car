import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 读取原始数据
train_path = 'data/used_car_train_20200313.csv'
test_path = 'data/used_car_testB_20200421.csv'
train_df = pd.read_csv(train_path, delim_whitespace=True)
test_df = pd.read_csv(test_path, delim_whitespace=True)

# 记录原始train/test的行数，方便后续合并处理
n_train = train_df.shape[0]
n_test = test_df.shape[0]

# 合并数据方便统一处理
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ========== 缺失值预处理 ==========
# 标记是否缺失的列（可选）
all_data['notRepairedDamage_missing'] = all_data['notRepairedDamage'].eq('-').astype(int)

# 填充缺失（例如用 'unknown' 填充）
all_data['notRepairedDamage'] = all_data['notRepairedDamage'].replace('-', 'unknown')

# 其他缺失字段填充（保守处理为-1）
all_data = all_data.fillna(-1)

# ========== 日期特征处理 ==========
all_data['regDate'] = pd.to_datetime(all_data['regDate'], format='%Y%m%d', errors='coerce')
all_data['creatDate'] = pd.to_datetime(all_data['creatDate'], format='%Y%m%d', errors='coerce')

all_data['regYear'] = all_data['regDate'].dt.year
all_data['regMonth'] = all_data['regDate'].dt.month
all_data['regDay'] = all_data['regDate'].dt.day
all_data['creatYear'] = all_data['creatDate'].dt.year
all_data['creatMonth'] = all_data['creatDate'].dt.month

# 精确计算使用年限（以月为单位）
all_data['carAge'] = ((all_data['creatDate'] - all_data['regDate']).dt.days / 365).clip(lower=0)

# 删除原始日期字段
all_data = all_data.drop(['regDate', 'creatDate'], axis=1)


# ========== 转换为数值类型 ==========
all_data['power'] = pd.to_numeric(all_data['power'], errors='coerce').fillna(-1)
all_data['kilometer'] = pd.to_numeric(all_data['kilometer'], errors='coerce').fillna(-1)
# ========== 衍生特征 ==========
# 每年行驶公里数
all_data['km_per_year'] = all_data['kilometer'] / (all_data['carAge'] + 0.1)  # 防止除0

# 处理power离群值，并构造对数特征
power_upper = 600
all_data['power'] = pd.to_numeric(all_data['power'], errors='coerce').fillna(-1)
all_data['power'] = np.clip(all_data['power'], 0, power_upper)
all_data['log_power'] = np.log1p(all_data['power'])

# regionCode频率编码
region_freq = all_data['regionCode'].value_counts(normalize=True)
all_data['regionCode_freq'] = all_data['regionCode'].map(region_freq)

# ========== 品牌/车型均值编码（仅使用训练集） ==========
# 确保品牌和车型字段为字符串类型（用于groupby）
train_df['brand'] = train_df['brand'].astype(str)
train_df['model'] = train_df['model'].astype(str)
all_data['brand'] = all_data['brand'].astype(str)
all_data['model'] = all_data['model'].astype(str)

# 计算训练集中的 brand/model 平均价格
brand_mean_price = train_df.groupby('brand')['price'].mean()
model_mean_price = train_df.groupby('model')['price'].mean()

# 映射到 all_data 中
all_data['brand_avg_price'] = all_data['brand'].map(brand_mean_price)
all_data['model_avg_price'] = all_data['model'].map(model_mean_price)

# 填补未知类别的均值（可能存在 test 中有 train 没见过的 brand/model）
all_data['brand_avg_price'] = all_data['brand_avg_price'].fillna(brand_mean_price.mean())
all_data['model_avg_price'] = all_data['model_avg_price'].fillna(model_mean_price.mean())



# ========== 类别变量处理 ==========
for col in all_data.columns:
    if all_data[col].dtype == 'object':
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

# ========== 删除无用字段 ==========

drop_cols = []
if 'SaleID' in all_data.columns:
    saleid = all_data['SaleID']
    drop_cols.append('SaleID')
else:
    saleid = None

if 'name' in all_data.columns:
    name = all_data['name']
    drop_cols.append('name')

all_data = all_data.drop(columns=drop_cols)


# ========== 拆分回训练集和测试集 ==========
train_fe = all_data.iloc[:n_train, :].copy()
test_fe = all_data.iloc[n_train:, :].copy()

# 恢复SaleID到test_fe，便于后续提交
# if saleid is not None:
#     test_fe['SaleID'] = saleid.iloc[n_train:].values
#     test_fe = test_fe[['SaleID'] + [col for col in test_fe.columns if col != 'SaleID']]
if saleid is not None:
    test_fe['SaleID'] = saleid.iloc[n_train:].values
if name is not None:
    test_fe['name'] = name.iloc[n_train:].values

# 可选：将这两个字段排在最前面（方便查看）
cols = test_fe.columns.tolist()
for col in ['SaleID', 'name']:
    if col in cols:
        cols.remove(col)
        cols = [col] + cols
test_fe = test_fe[cols]


pd.set_option('display.max_columns', None)
print(train_fe.head())
print(test_fe.head())

# ========== 保存为parquet格式 ==========
train_fe.to_parquet('train_fe.parquet', index=False)
test_fe.to_parquet('test_fe.parquet', index=False)

print('✅ 特征工程完成，已保存为 train_fe.parquet 和 test_fe.parquet')
