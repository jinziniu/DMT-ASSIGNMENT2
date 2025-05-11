# task4_LightGBM.py

import os
import pandas as pd
import lightgbm as lgb

# —— 1. 打印当前工作目录 & 文件，方便调试 —— #
print("工作目录:", os.getcwd())
print("目录下文件:", os.listdir(os.getcwd()))

# —— 2. 读取“老”预处理输出 —— #
# 训练/验证 切分后保存的
train_set = pd.read_csv("train_mod_train.csv")
val_set   = pd.read_csv("train_mod_val.csv")
# 完整 test 预处理结果
test_set  = pd.read_csv("test_set_VU_DM_modified.csv")

print("Train shape:", train_set.shape)
print("Val   shape:", val_set.shape)
print("Test  shape:", test_set.shape)

# —— 3. 定义特征列 —— #
exclude = ['srch_id','prop_id','click_bool','booking_bool','gross_bookings_usd','date_time']
feature_cols = [c for c in train_set.columns if c not in exclude]

# —— 3.1 只保留测试集中实际存在的列 —— #
feature_cols = [c for c in feature_cols if c in test_set.columns]
print("最终用于预测的特征：", feature_cols)

# 后面的 train_set[feature_cols] / test_set[feature_cols] 都不会报错了


# —— 4. 构造 LightGBM 排序 Dataset —— #
# group 信息：每个 srch_id 下候选数
train_group = train_set.groupby('srch_id').size().values
val_group   = val_set.groupby(  'srch_id').size().values

train_data = lgb.Dataset(
    train_set[feature_cols],
    label=train_set['click_bool'],
    group=train_group
)
valid_data = lgb.Dataset(
    val_set[feature_cols],
    label=val_set['click_bool'],
    group=val_group,
    reference=train_data
)

# —— 5. 训练参数 —— #
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
}


# —— 6. 训练 & 验证 —— #
gbm = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)




# —— 7. 对 test 做预测 —— #
preds = gbm.predict(test_set[feature_cols], num_iteration=gbm.best_iteration)

# —— 8. 生成提交文件 —— #
test_set['pred'] = preds
submission = (
    test_set
    .sort_values(['srch_id', 'pred'], ascending=[True, False])
    .loc[:, ['srch_id', 'prop_id']]
)
submission.to_csv("my_lgb_submission.csv", index=False)
print("Saved my_lgb_submission.csv")
