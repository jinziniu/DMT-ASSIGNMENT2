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

# 4. 构建 LGBM 排序数据集
train_group = train.groupby('srch_id').size().values
val_group   = val.groupby('srch_id').size().values
lgb_train = lgb.Dataset(train[features], label=train['relevance'], group=train_group)
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

# 8.5 分组 NDCG（Bias 检测）
def group_ndcg(dataset, model, k=5):
    mask_fam = dataset['srch_children_count'] > 0
    ndcg_fam = evaluate_ndcg(dataset[mask_fam], model)[k]
    ndcg_bus = evaluate_ndcg(dataset[~mask_fam], model)[k]
    return {'family': ndcg_fam, 'business': ndcg_bus}

baseline = group_ndcg(val, gbm, k=5)
print(f"Baseline group NDCG@5: {baseline}, Gap (family-business): {baseline['family'] - baseline['business']:.4f}")

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
