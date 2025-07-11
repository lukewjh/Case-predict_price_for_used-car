import pandas as pd, numpy as np, xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# ---------- 读取已完成的特征 ----------
train_df = pd.read_parquet("train_fe.parquet")
test_df  = pd.read_parquet("test_fe.parquet")

features = [c for c in train_df.columns if c not in ["price", "SaleID", "name"]]

X, y = train_df[features], train_df["price"]
X_test = test_df[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)


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


oof_xgb, oof_lgb, oof_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
pred_xgb, pred_lgb, pred_cat = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold+1} / 5 =====")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # --- XGB ---
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=1
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    pred_xgb += model_xgb.predict(X_test) / kf.n_splits

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=200)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration_)
    pred_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration_) / kf.n_splits

    # --- CatBoost ---
    model_cat = cb.CatBoostRegressor(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=1
    )
    oof_cat[val_idx] = model_cat.predict(X_val)
    pred_cat += model_cat.predict(X_test) / kf.n_splits



mae_xgb = mean_absolute_error(y, oof_xgb)
mae_lgb = mean_absolute_error(y, oof_lgb)
mae_cat = mean_absolute_error(y, oof_cat)
print(f"OOF 5-Fold MAE ➜  XGB: {mae_xgb:.2f} | LGB: {mae_lgb:.2f} | CAT: {mae_cat:.2f}")

# 简单平均
oof_mean   = (oof_xgb + oof_lgb + oof_cat) / 3
pred_mean  = (pred_xgb + pred_lgb + pred_cat) / 3
mae_mean   = mean_absolute_error(y, oof_mean)
print(f"✅ Ensemble OOF MAE: {mae_mean:.2f}")


submit = pd.DataFrame({
    "SaleID": test_df["SaleID"],
    "price" : pred_mean.round(2)
})
submit.to_csv("ensemble_xgb_lgb_cat.csv", index=False)
print("📝 已生成 ensemble_xgb_lgb_cat.csv")

