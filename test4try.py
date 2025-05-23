import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# 1. 读取数据
train = pd.read_csv('train_prepared.csv')
val   = pd.read_csv('validation_prepared.csv')
hold  = pd.read_csv('holdout_prepared.csv')
#  test_set_VU_DM.csv 运行相同的特征工程，生成：
test  = pd.read_csv('test_prepared.csv')
print(">>> Hold-out 样本数 (rows,cols):", hold.shape)
print(">>> Hold-out 列名：", hold.columns.tolist())
print(">>> Hold-out 前 5 行：")
print(hold.head())

# 2. 构造 Relevance 标签
for df in (train, val, hold):
    df['relevance'] = df['booking_bool'] * 5 + (df['click_bool'] - df['booking_bool'])

# 3. 确定特征列表
exclude = ['srch_id','prop_id','click_bool','booking_bool','gross_bookings_usd','date_time','relevance']
features = [c for c in train.columns if c not in exclude]
# 确保 test 含有相同特征
missing = set(features) - set(test.columns)
if missing:
    raise KeyError(f"缺失以下 test 特征: {missing}")

# 4. 计算样本权重（Re-weighting）
is_family_train = train['srch_children_count'] > 0
p_fam = is_family_train.mean()
weights = np.where(is_family_train, 1.0 / p_fam, 1.0)
# 归一化：保证平均权重为 1
weights = weights / np.mean(weights)
# Clip：避免过大或过小
weights = np.clip(weights, 0.5, 2.0)
# 对验证集也做同样的加权
is_family_val = val['srch_children_count'] > 0
val_weights = np.where(is_family_val, 1.0 / p_fam, 1.0)
val_weights = val_weights / np.mean(val_weights)
val_weights = np.clip(val_weights, 0.5, 2.0)

# 4. 构建 LGBM 排序数据集
train_group = train.groupby('srch_id').size().values
val_group   = val.groupby('srch_id').size().values
lgb_train = lgb.Dataset(train[features],
                        label=train['relevance'],
                        group=train_group,
                        weight=weights)
lgb_val   = lgb.Dataset(val[features],   label=val['relevance'],   group=val_group, reference=lgb_train)

# 5. LightGBM 参数
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1,3,5],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1
}

# 6. 训练 LambdaRank 模型 with callbacks
evals_result = {}
gbm = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train','val'],
    num_boost_round=1000,
    callbacks=[
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)
best_iter = gbm.best_iteration
print(f"Best iteration: {best_iter}")

hold['relevance'].value_counts()
print(hold['relevance'].value_counts())

# 7. 定义 NDCG 评估函数
def evaluate_ndcg(dataset, model):
    scores = {k: [] for k in (1,3,5)}
    for _, group in dataset.groupby('srch_id'):
        y_true = group['relevance'].values
        # if y_true.max() == 0:  # 全部都没有点击或预订
        #     scores[k].append(0.0)
        #     continue  # 跳过这组
        # y_pred = model.predict(group[features], num_iteration=best_iter)
        # for k in scores:
        #     scores[k].append(ndcg_score([y_true], [y_pred], k=k))
        y_pred = model.predict(group[features], num_iteration=best_iter)
        for k in scores:
           if y_true.max() == 0:
                   scores[k].append(0.0)
           else:
                  scores[k].append(ndcg_score([y_true], [y_pred], k=k))
    return {k: np.mean(v) for k, v in scores.items()}

# 8. 计算并打印 NDCG
val_ndcg = evaluate_ndcg(val, gbm)
hold_ndcg = evaluate_ndcg(hold, gbm)
print(f"Validation NDCG@1/3/5: {val_ndcg}")
print(f"Hold-out   NDCG@1/3/5: {hold_ndcg}")

# —— 插入开始 ——
# 按 srch_children_count 分组：>0 为家庭用户，=0 为非家庭用户
is_family_val = val['srch_children_count'] > 0

# 分别计算两组的 NDCG@5
ndcg_fam   = evaluate_ndcg(val[is_family_val], gbm)[5]
ndcg_non   = evaluate_ndcg(val[~is_family_val], gbm)[5]

# 打印群体差异
print(f"【偏差检测】验证集 NDCG@5 — 家庭: {ndcg_fam:.3f}, 非家庭: {ndcg_non:.3f}, 差异: {abs(ndcg_fam-ndcg_non):.3f}")
# —— 插入结束 ——


# 9. 输出并可视化特征重要性 Top10
imp = pd.DataFrame({'feature': features, 'gain': gbm.feature_importance('gain')})
imp = imp.sort_values('gain', ascending=False).head(10)
print("Top 10 feature importances:")
print(imp)

plt.figure(figsize=(8,6))
plt.barh(imp['feature'][::-1], imp['gain'][::-1])
plt.xlabel('Gain')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('lgbm_feature_importance.png')

# 10. 对 test 集预测并生成提交 Top5
test['pred'] = gbm.predict(test[features], num_iteration=best_iter)
submission = (
    test.sort_values(['srch_id','pred'], ascending=[True, False])
        .groupby('srch_id').head(5)[['srch_id','prop_id']]
)
submission.to_csv('lgbm_submission.csv', index=False)
print('Submission saved to lgbm_submission.csv')
