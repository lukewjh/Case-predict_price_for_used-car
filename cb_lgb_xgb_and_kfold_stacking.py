import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')



# 数据集准备
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载 训练集和测试集
def load_data():
    """
    加载原始数据
    """
    print("正在加载数据...")
    # 加载训练集
    train_data = pd.read_csv('data/used_car_train_20200313.csv', sep=' ')
    # 加载测试集
    test_data = pd.read_csv('data/used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    return train_data, test_data

# 开始做特征工程 1.合并训练、测试数据集 2.保存对应的SaleID 3.从训练集中获取对应的标签集
def preprocess_data(train_data, test_data):
    """
    数据预处理
    """
    print("\n开始数据预处理...")
    
    # 合并训练集和测试集进行特征工程
    train_data['source'] = 'train'
    test_data['source'] = 'test'
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 保存SaleID
    train_ids = train_data['SaleID']
    test_ids = test_data['SaleID']
    
    # 从训练集获取y值
    # y = train_data['price']
    y = np.log1p(train_data['price'])
    
    # 可视化对数变化后得分布
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(train_data['price'], bins=50)
    plt.title("原始价格分布")

    plt.subplot(1,2,2)
    sns.histplot(np.log1p(train_data['price']), bins=50)
    plt.title("对数变换后分布")

    plt.tight_layout()
    plt.show()

    return data, y, train_ids, test_ids

# 加载数据
train_data, test_data = load_data()

# 预处理数据
data, y_log, train_ids, test_ids = preprocess_data(train_data, test_data)

#时间相关的特征工程
def create_time_features(data):
    """
    创建时间特征
    """
    print("创建时间特征...")
    
    # 转换日期格式
    data['regDate'] = pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')
    data['creatDate'] = pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce')
    
    # 处理无效日期
    data.loc[data['regDate'].isnull(), 'regDate'] = pd.to_datetime('20160101', format='%Y%m%d')
    data.loc[data['creatDate'].isnull(), 'creatDate'] = pd.to_datetime('20160101', format='%Y%m%d')
    
    # 车辆年龄（天数）
    data['vehicle_age_days'] = (data['creatDate'] - data['regDate']).dt.days
    
    # 修复异常值
    data.loc[data['vehicle_age_days'] < 0, 'vehicle_age_days'] = 0
    
    # 车辆年龄（年）
    data['vehicle_age_years'] = data['vehicle_age_days'] / 365
    
    # 注册年份和月份
    data['reg_year'] = data['regDate'].dt.year
    data['reg_month'] = data['regDate'].dt.month
    data['reg_day'] = data['regDate'].dt.day
    
    # 创建年份和月份
    data['creat_year'] = data['creatDate'].dt.year
    data['creat_month'] = data['creatDate'].dt.month
    data['creat_day'] = data['creatDate'].dt.day
    
    # 是否为新车（使用年限<1年）
    data['is_new_car'] = (data['vehicle_age_years'] < 1).astype(int)
    
    # 季节特征
    data['reg_season'] = data['reg_month'].apply(lambda x: (x%12 + 3)//3)
    data['creat_season'] = data['creat_month'].apply(lambda x: (x%12 + 3)//3)
    
    # 每年行驶的公里数
    data['km_per_year'] = data['kilometer'] / (data['vehicle_age_years'] + 0.1)
    
    # 车龄分段
    data['age_segment'] = pd.cut(data['vehicle_age_years'], 
                                bins=[-0.01, 1, 3, 5, 10, 100], 
                                labels=['0-1年', '1-3年', '3-5年', '5-10年', '10年以上'])
    
    return data

# 创建时间特征
data = create_time_features(data)



# 创建车辆相关的特征
def create_car_features(data):
    """
    创建车辆特征
    """
    print("创建车辆特征...")
    
    # 缺失值处理
    numerical_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    for feature in numerical_features:
        # 标记缺失值
        data[f'{feature}_missing'] = data[feature].isnull().astype(int)
        # 填充缺失值
        data[feature] = data[feature].fillna(data[feature].median())
    
    # 将model转换为数值型特征
    data['model_num'] = data['model'].astype('category').cat.codes
    #data['model_num'] = data['model'].astype('int') # 不能这么写，因为有一个为空缺值
    
    # 品牌与车型组合
    data['brand_model'] = data['brand'].astype(str) + '_' + data['model'].astype(str)
        
    # 相对年份特征
    current_year = datetime.datetime.now().year
    data['car_age_from_now'] = current_year - data['reg_year']
    
    # 处理异常值
    numerical_cols = ['power', 'kilometer', 'v_0']
    for col in numerical_cols:
        Q1 = data[col].quantile(0.05)
        Q3 = data[col].quantile(0.95)
        IQR = Q3 - Q1
        data[f'{col}_outlier'] = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).astype(int)
        data[col] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    return data

# 创建车辆特征
data = create_car_features(data)


def create_statistical_features(data, train_idx):
    """
    创建统计特征
    """
    print("创建统计特征...")
    
    # 仅使用训练集数据创建统计特征
    train_data = data.iloc[train_idx].reset_index(drop=True)
    
    # 品牌级别统计
    brand_stats = train_data.groupby('brand').agg(
        brand_price_mean=('price', 'mean'),
        brand_price_median=('price', 'median'),
        brand_price_std=('price', 'std'),
        brand_price_count=('price', 'count')
    ).reset_index()
    

    # 合并统计特征
    data = data.merge(brand_stats, on='brand', how='left')
    
    # 相对价格特征（相对于平均价格）
    data['brand_price_ratio'] = data['brand_price_mean'] / data['brand_price_mean'].mean()
    
    return data

# 找回训练集的索引
train_idx = data[data['source'] == 'train'].index
test_idx = data[data['source'] == 'test'].index

# 创建统计特征
data = create_statistical_features(data, train_idx)

#找到适用于catboost的特征并单独存放
def encode_categorical_features(data):
    """
    编码分类特征
    """
    print("编码分类特征...")
    
    # 目标编码的替代方案 - 频率编码
    categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_cols:
        # 填充缺失值
        data[col] = data[col].fillna('未知')
        
        # 频率编码
        freq_encoding = data.groupby(col).size() / len(data)
        data[f'{col}_freq'] = data[col].map(freq_encoding)
    
    # 将分类变量转换为CatBoost可以识别的格式
    for col in categorical_cols:
        data[col] = data[col].astype('str')
    
    return data, categorical_cols

# 编码分类特征
data, categorical_cols = encode_categorical_features(data)

#最终特征选择
def feature_selection(data, categorical_cols):
    """
    特征选择和最终数据准备
    """
    print("特征选择和最终数据准备...")
    
    # 删除不再需要的列, 所有车offerType=0,seller只有1个为1，其他都为0
    drop_cols = ['regDate', 'creatDate', 'price', 'SaleID', 'name', 'offerType', 'seller', 'source']
    data = data.drop(drop_cols, axis=1, errors='ignore')
    
    # 确保所有分类特征都被正确标记
    # 添加age_segment到分类特征列表中
    if 'age_segment' not in categorical_cols and 'age_segment' in data.columns:
        categorical_cols.append('age_segment')
    
    # 确保brand_model也被标记为分类特征
    if 'brand_model' not in categorical_cols and 'brand_model' in data.columns:
        categorical_cols.append('brand_model')
    
    # 转换分类特征
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    return data, categorical_cols

# 特征选择和最终准备
data, cat_features = feature_selection(data, categorical_cols)

#分离训练集和测试集
X_train_full = data.iloc[train_idx].reset_index(drop=True)
X_test = data.iloc[test_idx].reset_index(drop=True)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_log, test_size=0.1, random_state=42
)






# 公共的多折训练函数
n_splits = 5
price_bins = pd.qcut(y_log, 5, labels=False)   # 与 CatBoost 时相同
skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
from sklearn.preprocessing import OrdinalEncoder
import gc
def encode_categorical_once(X_train, X_test, cat_cols):
    """
    使用 OrdinalEncoder 一次性 fit+transform。
    返回 (X_train_enc, X_test_enc)
    """
    oe = OrdinalEncoder(handle_unknown='use_encoded_value',
                        unknown_value=-1, dtype='int64')
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    X_train_enc[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test_enc[cat_cols]  = oe.transform(X_test[cat_cols])
    return X_train_enc, X_test_enc


from sklearn.preprocessing import LabelEncoder
def encode_categorical(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def kfold_train_predict(model_cls, param, X, y_log, X_test, cat_features=None):
    if model_cls.__name__ == 'XGBRegressor' and cat_features:
        X, X_test = encode_categorical_once(X, X_test, cat_features)


    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for fold,(tr,val) in enumerate(skf.split(X, price_bins), 1):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y_log.iloc[tr], y_log.iloc[val]

        model = model_cls(**param)
        

        if model_cls.__name__ == 'CatBoostRegressor':
            model.fit(Pool(X_tr, y_tr, cat_features),
                      eval_set=Pool(X_val, y_val, cat_features),
                      use_best_model=True, early_stopping_rounds=200, verbose=1)
        elif model_cls.__name__ == "LGBMRegressor":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(100)
                ]
            )
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=200
                      )

        oof[val] = model.predict(X_val)
        test_pred += model.predict(X_test) / n_splits
    return oof, test_pred


# CatBoost 已有 params_cb

# ========= CatBoost =========
params_cb = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=8,                    # ≈ XGB max_depth, LGB 叶子数=256
    l2_leaf_reg=5,              # ≈ reg_lambda
    bootstrap_type="Bayesian",             # 行抽样
    #rsm=0.8,                    # 列抽样
    bagging_temperature=0.5,
    random_seed=42,
    one_hot_max_size=5,
    od_type="Iter",
    od_wait=200,
    loss_function="MAE",
    eval_metric="MAE",
    task_type="GPU",
    thread_count=-1,
    verbose=200
)

# ========= LightGBM =========
params_lgb = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=256,             # 2^depth ≈ 256 when depth=8
    subsample=0.8,              # bagging_fraction
    colsample_bytree=0.8,       # feature_fraction
    reg_lambda=1.0,
    objective="mae",
    random_state=42,          
)

# ========= XGBoost =========
params_xgb = dict(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:absoluteerror",
    eval_metric='mae',
    random_state=42,
    early_stopping_rounds=200,
    tree_method="hist"          # CPU 快速版
)


try:
    xgb_oof, xgb_test = kfold_train_predict(xgb.XGBRegressor, params_xgb,
                                            X_train_full, y_log, X_test, cat_features)
    print("XGBoost done")

    cb_oof,  cb_test  = kfold_train_predict(CatBoostRegressor, params_cb,
                                            X_train_full, y_log, X_test, cat_features)
    print("CatBoost done")
    
    lgb_oof, lgb_test = kfold_train_predict(lgb.LGBMRegressor, params_lgb,
                                            X_train_full, y_log, X_test)
    print("LightGBM done")
    
except Exception as e:
    print("Error during model training:", str(e))
    raise

print("训练完成-----------------")

#然后组装二级训练集
meta_X = pd.DataFrame({
    'cb' :  cb_oof,
    'lgb':  lgb_oof,
    'xgb':  xgb_oof,
    # 也可加 rank 特征让 meta 学排序
    # rank 需要 Series
    'cb_r' : pd.Series(cb_oof).rank(pct=True),
    'lgb_r': pd.Series(lgb_oof).rank(pct=True),
    'xgb_r': pd.Series(xgb_oof).rank(pct=True),
})
meta_test = pd.DataFrame({
    'cb' :  cb_test,
    'lgb':  lgb_test,
    'xgb':  xgb_test,
    'cb_r' : pd.Series(cb_test).rank(pct=True),
    'lgb_r': pd.Series(lgb_test).rank(pct=True),
    'xgb_r': pd.Series(xgb_test).rank(pct=True),
})

from sklearn.linear_model import RidgeCV
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(meta_X, y_log)          # 仍在 log 空间
meta_oof_log = meta_model.predict(meta_X)
meta_test_log = meta_model.predict(meta_test)



mae_meta = mean_absolute_error(np.expm1(y_log), np.expm1(meta_oof_log))
print(f"Stacking OOF MAE = {mae_meta:.2f}")

submit_df = pd.DataFrame({
    'SaleID': test_ids,
    'price' : np.expm1(meta_test_log)
})
submit_df.to_csv('stacked_submit.csv', index=False)

