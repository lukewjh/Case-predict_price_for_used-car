import pandas as pd, numpy as np, xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------- 读取特征数据 ----------
train_df = pd.read_parquet("train_fe.parquet")
test_df  = pd.read_parquet("test_fe.parquet")

features = [c for c in train_df.columns if c not in ["price", "SaleID", "name"]]
X, y = train_df[features], train_df["price"]
X_test = test_df[features]

# 可选：划分一部分验证集用于评估（比如 10%）
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ---------- 各模型参数 ----------
xgb_params = dict(
    objective="reg:absoluteerror",
    eval_metric="mae",
    tree_method="hist",
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=3000,
    reg_lambda=2.0,
    random_state=42
)

lgb_params = dict(
    objective="mae",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=256,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=5000,
    reg_lambda=1.0,
    random_state=42
)

cat_params = dict(
    loss_function="MAE",
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    iterations=5000,
    random_seed=42,
    verbose=False
)

# ---------- 模型训练 ----------
print("\n🚀 Training XGBoost...")
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1)
pred_xgb = model_xgb.predict(X_test)
val_pred_xgb = model_xgb.predict(X_val)

print("\n🚀 Training LightGBM...")
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="mae", callbacks=[lgb.early_stopping(200)])
pred_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration_)
val_pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration_)

print("\n🚀 Training CatBoost...")
model_cat = cb.CatBoostRegressor(**cat_params)
model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=1)
pred_cat = model_cat.predict(X_test)
val_pred_cat = model_cat.predict(X_val)

# ---------- 验证评估 ----------
mae_xgb = mean_absolute_error(y_val, val_pred_xgb)
mae_lgb = mean_absolute_error(y_val, val_pred_lgb)
mae_cat = mean_absolute_error(y_val, val_pred_cat)
print(f"\n📊 Validation MAE ➜  XGB: {mae_xgb:.2f} | LGB: {mae_lgb:.2f} | CAT: {mae_cat:.2f}")

# ---------- 简单平均集成 ----------
pred_mean = (pred_xgb + pred_lgb + pred_cat) / 3
val_pred_mean = (val_pred_xgb + val_pred_lgb + val_pred_cat) / 3
mae_ensemble = mean_absolute_error(y_val, val_pred_mean)
print(f"✅ Ensemble Validation MAE: {mae_ensemble:.2f}")

# ---------- 生成提交文件 ----------
submit = pd.DataFrame({
    "SaleID": test_df["SaleID"],
    "price": pred_mean.round(2)
})
submit.to_csv("ensemble_xgb_lgb_cat.csv", index=False)
print("📝 已生成 ensemble_xgb_lgb_cat.csv")
