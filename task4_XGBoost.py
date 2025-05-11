import os
import pandas as pd
import xgboost as xgb


def load_data(train_path, val_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    return train, val


def get_feature_columns(df, exclude_cols):
    return [c for c in df.columns if c not in exclude_cols]


def prepare_dmatrix(df, feature_cols, label_col=None, group_col=None):
    # 构造 DMatrix
    data = df[feature_cols]
    label = df[label_col] if label_col is not None else None
    dmat = xgb.DMatrix(data=data, label=label)
    # 设置分组
    if group_col:
        group_sizes = df.groupby(group_col).size().tolist()
        dmat.set_group(group_sizes)
    return dmat


def train_model(dtrain, dvalid, params, num_rounds=500, early_stop=30):
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=early_stop,
        verbose_eval=50
    )
    return bst


def predict_and_save(bst, test_df, feature_cols, output_path):
    # 补齐缺失特征
    missing = [c for c in feature_cols if c not in test_df.columns]
    for col in missing:
        test_df[col] = 0  # 默认填 0，可根据需求修改

    # 对齐顺序
    dtest = xgb.DMatrix(test_df[feature_cols])
    preds = bst.predict(dtest)
    test_df['pred'] = preds

    submission = (
        test_df
        .sort_values(['srch_id', 'pred'], ascending=[True, False])
        .loc[:, ['srch_id', 'prop_id']]
    )
    submission.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    # 1. 工作目录 & 文件
    print("工作目录:", os.getcwd())
    print("目录文件:", os.listdir(os.getcwd()))

    # 2. 读取数据
    train_set, val_set = load_data(
        "train_mod_train.csv",
        "train_mod_val.csv"
    )
    print("Train shape:", train_set.shape)
    print("Val   shape:", val_set.shape)

    # 3. 特征列（排除无用列）
    exclude = [
        'srch_id', 'prop_id', 'click_bool', 'booking_bool',
        'gross_bookings_usd', 'date_time'
    ]
    feature_cols = get_feature_columns(train_set, exclude)
    print("Feature count:", len(feature_cols))

    # 4. 构造 DMatrix 并设置 group
    dtrain = prepare_dmatrix(
        train_set, feature_cols,
        label_col='click_bool', group_col='srch_id'
    )
    dvalid = prepare_dmatrix(
        val_set, feature_cols,
        label_col='click_bool', group_col='srch_id'
    )

    # 5. XGBoost 参数
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@5',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 1,
        'seed': 42
    }

    # 6. 训练
    bst = train_model(dtrain, dvalid, params)

    # 7. 预测 & 保存
    test_set = pd.read_csv("test_set_VU_DM_modified.csv")
    predict_and_save(bst, test_set, feature_cols, "my_xgb_submission.csv")


if __name__ == "__main__":
    main()
