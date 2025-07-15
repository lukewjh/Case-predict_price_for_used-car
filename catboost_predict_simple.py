import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ───────────────────────────────────────────────────────────
# 1. 造一份玩具数据：包含数值 + 类别列
# ───────────────────────────────────────────────────────────
n = 1000
rng = np.random.RandomState(42)

data = pd.DataFrame({
    'brand':  rng.choice(['BMW', 'Audi', 'Toyota', 'Tesla'], n),
    'gearbox': rng.choice(['AT', 'MT'], n),
    'power':   rng.normal(150, 30, n).clip(50, 300),
    'kilometer': rng.uniform(0, 15, n),
})
# 目标：根据品牌、变速箱、功率、公里数 → 估价 price
data['price'] = (
      20000
    + data['power'] * 120
    - data['kilometer'] * 800
    + data['brand'].map({'BMW':5000, 'Audi':3000, 'Toyota':0, 'Tesla':8000})
    + rng.normal(0, 2000, n)         # 噪声
)

X_train, X_val, y_train, y_val = train_test_split(
    data.drop('price', axis=1),
    data['price'],
    test_size=0.2,
    random_state=42
)

# ───────────────────────────────────────────────────────────
# 2. 指定类别列，构建 Pool
# ───────────────────────────────────────────────────────────
cat_features = ['brand', 'gearbox']

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

# ───────────────────────────────────────────────────────────
# 3. 建模与训练（含早停）
# ───────────────────────────────────────────────────────────
model = CatBoostRegressor(
    iterations=2000,          # 最多 2000 棵树
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='MAE',
    random_seed=42,
    verbose=200,              # 每 200 轮打印一次
    early_stopping_rounds=100 # 100 轮无提升就停止
)

model.fit(train_pool, eval_set=val_pool)

# ───────────────────────────────────────────────────────────
# 4. 预测 + 评估
# ───────────────────────────────────────────────────────────
pred_val = model.predict(X_val)
print(f"MAE  : {mean_absolute_error(y_val, pred_val):,.0f}")
print(f"R²   : {r2_score(y_val, pred_val):.3f}")

# ───────────────────────────────────────────────────────────
# 5. 保存 / 加载
# ───────────────────────────────────────────────────────────
model.save_model("catboost_regressor.cbm")
# model = CatBoostRegressor().load_model("catboost_regressor.cbm")
