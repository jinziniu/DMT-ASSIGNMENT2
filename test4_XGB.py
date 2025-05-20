# test4.py: 基于 test3_new.py 预处理结果的 XGBoost 排序示例
import pandas as pd
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score

# 1. 加载 test3 预处理后输出的文件
train_df = pd.read_csv('train_prepared.csv', parse_dates=['date_time'])
val_df   = pd.read_csv('validation_prepared.csv', parse_dates=['date_time'])
test_df  = pd.read_csv('test_prepared.csv', parse_dates=['date_time'])

# 2. 构造排序标签：booking_bool * 5 + click_bool
for df in [train_df, val_df]:
    df['label'] = df['booking_bool'] * 5 + df['click_bool']

# 3. 定义特征列（排除非特征列）
exclude_cols = [
    'srch_id', 'prop_id', 'date_time',  # 分组与标识
    'booking_bool', 'click_bool',        # 原始标签
    'label'                              # 排序目标
]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# 4. 准备训练/验证数据及分组信息
X_train = train_df[feature_cols]
y_train = train_df['label']
group_train = train_df.groupby('srch_id').size().to_list()

X_val   = val_df[feature_cols]
y_val   = val_df['label']
group_val = val_df.groupby('srch_id').size().to_list()

# 5. 初始化并训练 XGBRanker（不带早停）
ranker = XGBRanker(
    objective='rank:ndcg',
    eval_metric='ndcg@5',
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)
# 直接训练，无 early_stopping 周期设置
ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    verbose=True
)

# 6. 本地验证集 NDCG@5 评估
val_scores = []
for _, grp in val_df.groupby('srch_id'):
    y_true = grp['label'].values
    y_pred = ranker.predict(grp[feature_cols])
    val_scores.append(ndcg_score([y_true], [y_pred], k=5))
avg_ndcg5 = sum(val_scores) / len(val_scores)
print(f'Local validation NDCG@5: {avg_ndcg5:.4f}')

# 7. 测试集预测与提交文件
X_test = test_df[feature_cols]
test_df['rank_score'] = ranker.predict(X_test)
submission = (
    test_df
    .sort_values(['srch_id', 'rank_score'], ascending=[True, False])
    .groupby('srch_id')
    .head(5)
)
submission[['srch_id', 'prop_id']].to_csv('submission_xgb_rank.csv', index=False)
print('Submission file saved to submission_xgb_rank.csv')
