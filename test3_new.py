import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # （1）丢弃训练/预测时不能用的泄漏列
    drop_cols = [
        'gross_bookings_usd', 'position'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 1. 缺失列分类（阈值95%）
    very_high_na = [c for c in df.columns if df[c].isnull().mean() > 0.95]
    moderate_na = [c for c in df.columns if 0 < df[c].isnull().mean() <= 0.95]
    # 非常高缺失：保留指示列，删除原列
    for c in very_high_na:
        df[f'{c}_isna'] = df[c].isnull().astype('int8')
        df = df.drop(columns=[c])
    # 中度缺失：生成指示列，显式填充后保留原列
    for c in moderate_na:
        df[f'{c}_isna'] = df[c].isnull().astype('int8')
        df[c] = df[c].fillna(df[c].median()) if df[c].dtype.kind in 'fiu' else df[c].fillna(df[c].mode().iloc[0])

    # 2. 对数变换 + 标准化
    skew_cols = ['price_usd', 'orig_destination_distance', 'visitor_hist_adr_usd', 'prop_log_historical_price']
    exist_skew = [c for c in skew_cols if c in df.columns]
    for c in exist_skew:
        df[c] = np.log1p(df[c])
    if exist_skew:
        df[exist_skew] = StandardScaler().fit_transform(df[exist_skew])

    # 3. 分箱 & One-Hot 编码
    bin_cols = [c for c in ['price_usd', 'prop_review_score', 'visitor_hist_adr_usd'] if c in df.columns]
    for c in bin_cols:
        df[f'{c}_bin'] = pd.qcut(df[c], 10, labels=False, duplicates='drop')
    if bin_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        binned = ohe.fit_transform(df[[f'{c}_bin' for c in bin_cols]])
        df = pd.concat([df, pd.DataFrame(binned, columns=ohe.get_feature_names_out(), index=df.index)], axis=1)
        df = df.drop(columns=[f'{c}_bin' for c in bin_cols])

    # 4. 组内统计与排名
    grp = df.groupby('srch_id')
    df['price_min']      = grp['price_usd'].transform('min')
    df['price_rank']     = grp['price_usd'].rank(method='dense')
    df['price_pct_diff'] = (df['price_usd'] - df['price_min']) / (df['price_min'] + 1e-6)
    if 'prop_review_score' in df.columns:
        df['review_rank'] = grp['prop_review_score'].rank(ascending=False, method='dense')

    #price_rank 分布
    plt.figure(figsize=(8, 5))
    plt.hist(df['price_rank'].dropna(), bins=50)
    plt.xlabel('Price Rank')
    plt.ylabel('Frequency')
    plt.title('Distribution of Price Rank')
    plt.tight_layout()
    plt.show()



    # 5. 交互特征
    df['loc_stay_inter'] = df['prop_location_score1'] * df['srch_length_of_stay']
    if 'visitor_hist_adr_usd' in df.columns:
        df['hist_ratio'] = df['price_usd'] / (df['visitor_hist_adr_usd'] + 1e-6)

    # 2) Hist Ratio distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['hist_ratio'].dropna(), bins=50)
    plt.xlabel('Hist Ratio (price_usd / visitor_hist_adr_usd)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hist Ratio')
    plt.tight_layout()
    plt.show()

    # 6. 时间特征
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour']     = df['date_time'].dt.hour
    df['weekday']  = df['date_time'].dt.weekday
    return df


def split_holdout(df: pd.DataFrame, cutoff_date: str = '2013-06-01'):
    cutoff = pd.to_datetime(cutoff_date)
    train_all = df[df['date_time'] < cutoff].copy()
    holdout   = df[df['date_time'] >= cutoff].copy()
    return train_all, holdout


def split_train_validation(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # 基于 srch_id 分组抽样，训练/验证再拆分
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df['srch_id']))
    train = df.iloc[train_idx].copy()
    val   = df.iloc[val_idx].copy()
    return train, val


def main():
    df = pd.read_csv('training_set_VU_DM.csv', parse_dates=['date_time'])
    df_prepped = prepare_features(df)
    print("数据 date_time 最早：", df_prepped['date_time'].min())
    print("数据 date_time 最晚：", df_prepped['date_time'].max())
    # 先按时间切分留存集
    train_all, holdout = split_holdout(df_prepped)
    # 再对训练集做组内 train/validation 拆分
    train, val = split_train_validation(train_all)
    # 保存
    train.to_csv('train_prepared.csv', index=False)
    val.to_csv('validation_prepared.csv', index=False)
    holdout.to_csv('holdout_prepared.csv', index=False)
    print('Data preparation and splitting completed.')
    print(">>> 重新 split 后，holdout 样本数：", holdout.shape)

    # 2) 测试集特征处理
    test_path = 'test_set_VU_DM.csv'  # 根据实际文件名修改
    test_df = pd.read_csv(test_path, parse_dates=['date_time'])
    test_prepped = prepare_features(test_df)

    # ---- 兼容性修补 ----
    # 如果 test 缺少 position，则补为 0
    # if 'position' not in test_prepped.columns:
    #     test_prepped['position'] = 0
    # # 补齐训练集中生成的 gross_bookings_usd_isna
    # if 'gross_bookings_usd_isna' not in test_prepped.columns:
    #     test_prepped['gross_bookings_usd_isna'] = 1
    for col in [c for c in df.columns if c.endswith('_isna')]:
        if col not in test_prepped.columns:
            test_prepped[col] = 1

    test_prepped.to_csv('test_prepared.csv', index=False)
    print('Test set prepared and saved as test_prepared.csv.')

if __name__ == '__main__':
    main()
