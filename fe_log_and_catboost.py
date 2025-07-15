#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
二手车价格预测 - 高级特征工程与CatBoost建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

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
data, y, train_ids, test_ids = preprocess_data(train_data, test_data)
data


# In[2]:

# 特征工程1 汽车信息的时间相关的特征工程处理
# 原始数据集中的时间有 汽车注册时间、创建这条二手车记录的时间  
# 使用创建这条二手车记录的减去汽车注册时间 就是该车的年龄 并且如果有 异常值（年龄数小于0） 就将该值设置为0  vehicle_age_days
# 并将这个年龄除365得出对应的年数 vehicle_age_years
# 将车的注册时间转为三个字段 年、月、日 reg_year reg_month reg_day
# 将车的二手车记录创建时间转为 年、月、日 creat_year creat_month creat_day
# 创建新字段：通过vehicle_age_years字段是否小于1 推断是否是新车
#  创建该车注册时间的季节字段并计算
#  创建该车二手车记录创建时间的季节字段并计算
# 使用已行驶的公里数和汽车年龄创建每年的行驶公里数字段 km_per_year
# 创建车年龄分段的字段
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
data[['regDate', 'creatDate', 'reg_year', 'reg_month', 'reg_day', 'creat_year', 'creat_month', 'creat_day', 
      'vehicle_age_days', 'vehicle_age_years', 'is_new_car', 'reg_season', 'creat_season', 'km_per_year', 'age_segment']]


# In[3]:

# 创建车辆相关的特征
# power、kilometer、v_0-v_14的匿名特征的的缺失值处理  
#    在这条数据上创建一个是否缺失的标识 XXX_missing astype把True、False转为 0或1
#    并且填充该字段列的中位数
# 创建model字段 并且转为model_num分类 用数值型表示
# 组合model和brand 车型和品牌强关联 catboost支持直接传入字符串的并且自动分类（✅ 原生支持（内部做 hashing + CTR 处理））
# 计算现在的时间到车注册的时间 创建车从创建至今的年龄 car_age_from_now 
# 处理 power、kilometer、v_0的异常值
# 
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
data


# In[4]:

# 创建相关统计特征
#  仅在训练集上创建品牌与价格关系的统计特征
#  创建新字段 brand_price_mean brand_price_median brand_price_std（该品牌成交价的 标准差（价格波动大小）） 
#  brand_price_count brand_price_ratio

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
data


# In[5]:

# 创建'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage'的频率字段
# 然后将这些字段值转为str 适用于CatBoost
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
data


# In[6]:


categorical_cols


# In[7]:

#删除无用的字段列 ['regDate', 'creatDate', 'price', 'SaleID', 'name', 'offerType', 'seller', 'source']
# 检查之前创建的字段是否在分类特征中
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
data


# In[11]:


#data[['offerType', 'seller']]
#data['offerType'].value_counts()
#data['seller'].value_counts()


# In[13]:


# 分离训练集和测试集
X_train_full = data.iloc[train_idx].reset_index(drop=True)
X_test = data.iloc[test_idx].reset_index(drop=True)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y, test_size=0.1, random_state=42
)

# 保存处理后的数据
joblib.dump(X_train, 'processed_data/fe_X_train.joblib')
joblib.dump(X_val, 'processed_data/fe_X_val.joblib')
joblib.dump(y_train, 'processed_data/fe_y_train.joblib')
joblib.dump(y_val, 'processed_data/fe_y_val.joblib')
joblib.dump(X_test, 'processed_data/fe_test_data.joblib')
joblib.dump(test_ids, 'processed_data/fe_sale_ids.joblib')
joblib.dump(cat_features, 'processed_data/fe_cat_features.joblib')

print("预处理后的数据已保存")


# In[9]:


def train_catboost_model(X_train, X_val, y_train, y_val, cat_features):
    """
    训练CatBoost模型
    """
    print("\n开始训练CatBoost模型...")
    
    # 创建数据池
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # 设置模型参数
    params = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 100,
        'verbose': 100,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1
    }
    
    # 创建模型
    model = CatBoostRegressor(**params)
    


    # 训练模型
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=True
    )
    
    # 保存模型
    model.save_model('processed_data/fe_catboost_model.cbm')
    print("模型已保存到 processed_data/fe_catboost_model.cbm")
    
    return model
# model = train_catboost_model(X_train, X_val, y_train, y_val, cat_features)

from sklearn.model_selection import StratifiedKFold
import numpy as np
from catboost import CatBoostRegressor, Pool
# 使用交叉验证的方式训练模型
def train_catboost_kfold(X_train_full, y_log, X_test, cat_features,
                         n_splits=5, base_seed=42, params=None):
    """
    KFold + 多随机种子训练 CatBoost
    ----------
    返回：
        oof_pred       : 与 y_log 同长度的 OOF 预测（log 空间）
        test_pred_mean : 测试集平均预测（log 空间，可 np.expm1() 反变换）
        models         : 每折训练好的 CatBoost 模型列表
    """
    if params is None:
        params = {
            'iterations'    : 3000,
            'learning_rate' : 0.03,
            'depth'         : 6,
            'l2_leaf_reg'   : 3,
            'bootstrap_type': 'Bayesian',
            'od_type'       : 'Iter',
            'od_wait'       : 200,
            'loss_function' : 'MAE',
            'eval_metric'   : 'MAE',
            'task_type'     : 'CPU',
            'thread_count'  : -1,
            'verbose'       : 200
        }

    # ① 生成分箱标签（5 等分可调）
    price_bins = pd.qcut(y_log, q=5, labels=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    oof_pred   = np.zeros(len(X_train_full))
    test_pred  = np.zeros(len(X_test))
    models     = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, price_bins), 1):
        print(f"\n▶▶ Fold {fold}/{n_splits}")

        X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx],      y_log.iloc[val_idx]

        # ② 每折换随机种子
        params['random_seed'] = base_seed + fold

        model = CatBoostRegressor(**params)

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool   = Pool(X_val, y_val, cat_features=cat_features)

        model.fit(train_pool,
                  eval_set=val_pool,
                  use_best_model=True,
                  early_stopping_rounds=200)

        # OOF
        oof_pred[val_idx] = model.predict(val_pool)

        # Test 预测累加
        test_pred += model.predict(Pool(X_test, cat_features=cat_features)) / n_splits

        models.append(model)

    # 保存第一折模型示例（可按需保存全部）
    models[0].save_model('processed_data/fe_catboost_model_fold1.cbm')
    print("★ KFold 训练完成 —— 已保存 fold1 模型")

    return oof_pred, test_pred, models

# 训练CatBoost模型
# y 已经是 log1p(price) 的 Series
oof_log, test_log, model = train_catboost_kfold(
    X_train_full=X_train_full,
    y_log=y,                 # log 标签
    X_test=X_test,
    cat_features=cat_features,
    n_splits=5,
    base_seed=42
)

# 反变换评估 OOF
from sklearn.metrics import mean_absolute_error
mae_kfold = mean_absolute_error(np.expm1(y), np.expm1(oof_log))
print(f"KFold OOF MAE = {mae_kfold:.2f}")
# 生成提交
submit_df = pd.DataFrame({
    'SaleID': test_ids,
    'price' : np.expm1(test_log)   # 反变换
})
submit_df.to_csv('fe_catboost_kfold_submit.csv', index=False)

# In[10]:


def evaluate_model(model, X_val, y_val, cat_features):
    """
    评估模型性能
    """
    # 创建验证数据池
    val_pool = Pool(X_val, cat_features=cat_features)
    
    # 预测
    y_pred = model.predict(val_pool)
    # 反变换回原始单位
    y_pred = np.expm1(y_pred)   # inverse of log1p
    y_val_true = np.expm1(y_val)
    
    # 计算评估指标
    mse = mean_squared_error(y_val_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_true, y_pred)
    r2 = r2_score(y_val_true, y_pred)


    # mse = mean_squared_error(y_val, y_pred)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_val, y_pred)
    # r2 = r2_score(y_val, y_pred)
    
    print("\n模型评估结果：")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R2分数: {r2:.4f}")
    
    # 绘制预测值与实际值的对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('CatBoost预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('fe_catboost_prediction_vs_actual.png')
    plt.close()
    
    return rmse, mae, r2

# 评估模型
# rmse, mae, r2 = evaluate_model(model, X_val, y_val, cat_features)


# In[11]:


def plot_feature_importance(model, X_train):
    """
    绘制特征重要性图
    """
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 保存特征重要性到CSV
    importance_df.to_csv('fe_catboost_feature_importance.csv', index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('CatBoost Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('fe_catboost_feature_importance.png')
    plt.close()
    
    return importance_df
    
# 绘制特征重要性
# importance_df = plot_feature_importance(model, X_train)


# In[14]:


def predict_test_data(model, X_test, test_ids, cat_features):
    """
    预测测试集数据
    """
    print("\n正在预测测试集...")
    
    # 创建测试数据池
    test_pool = Pool(X_test, cat_features=cat_features)
    
    # 预测
    predictions = model.predict(test_pool)
    predictions = np.expm1(predictions)  # 反变换
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': test_ids,
        'price': predictions
    })
    
    # 保存预测结果
    submit_data.to_csv('fe_catboost_submit_result.csv', index=False)
    print("预测结果已保存到 fe_catboost_submit_result.csv")

# 预测测试集
# predict_test_data(model, X_test, test_ids, cat_features)

# print("\n模型训练、评估和预测完成！")
# print(f"Top 10 重要特征:\n{importance_df.head(10)}")


# In[37]:


# X_test['brand_model'].isnull().sum()
# cat_features
# X_test['vehicle_age_years'].describe()
# #X_test['vehicle_age_years'].isnull().sum()
# data.loc[data['age_segment'].isnull(), 'vehicle_age_years']


# In[12]:


# # 车龄分段
# data['age_segment'] = pd.cut(data['vehicle_age_years'], 
#                             bins=[-0.01, 1, 3, 5, 10, 100], 
#                             labels=['0-1年', '1-3年', '3-5年', '5-10年', '10年以上'])


# In[16]:


#cat_features

