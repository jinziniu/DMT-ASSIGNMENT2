# task2_data_preparation_modified.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def reduce_mem(df):
    int_cols = df.select_dtypes(include=['int64']).columns
    float_cols = df.select_dtypes(include=['float64']).columns
    df[int_cols]   = df[int_cols].apply(pd.to_numeric, downcast='integer')
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    return df

def preprocess(df):
    df = clean_columns(df)
    df = reduce_mem(df)

    # --- 缺失 & 标记 ---
    comp_rate_cols = [c for c in df.columns if c.startswith('comp') and 'rate' in c]
    df['has_comp_rate'] = df[comp_rate_cols].notna().any(axis=1).astype('int8')
    df['hist_exists']   = df['visitor_hist_starrating'].notna().astype('int8')

    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(0)
    df['visitor_hist_adr_usd']    = df['visitor_hist_adr_usd'].fillna(0)
    df['prop_location_score2']    = df['prop_location_score2'].fillna(df['prop_location_score2'].median())
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].median())
    df['prop_review_score']       = df['prop_review_score'].fillna(df['prop_review_score'].median())

    # --- 连续变换 & 分箱 ---
    df['log_price']    = np.log1p(df['price_usd'])
    df['log_hist_adr'] = np.log1p(df['visitor_hist_adr_usd'])
    df['price_qtile']  = pd.qcut(df['price_usd'], 10, labels=False, duplicates='drop').astype('int8')

    # --- 排序 & 交互 ---
    df['price_rank']  = df.groupby('srch_id')['price_usd'].rank(method='dense').astype('int16')
    df['score1_rank'] = df.groupby('srch_id')['prop_location_score1'].rank(ascending=False, method='dense').astype('int16')
    df['diff_starr']  = df['prop_starrating'] - df['visitor_hist_starrating']
    df['diff_price']  = df['price_usd'] - df['visitor_hist_adr_usd']
    df['same_country']= (df['visitor_location_country_id']==df['prop_country_id']).astype('int8')

    # --- 时间特征 ---
    df['date_time']   = pd.to_datetime(df['date_time'])
    df['hour']        = df['date_time'].dt.hour.astype('int8')
    df['wday']        = df['date_time'].dt.weekday.astype('int8')
    df['month']       = df['date_time'].dt.month.astype('int8')
    df['is_weekend']  = (df['wday']>=5).astype('int8')
    df['daypart']     = (df['hour']//6).astype('int8')

    # --- 离散转 category ---
    for c in ['prop_starrating','srch_length_of_stay','srch_adults_count',
              'srch_children_count','daypart','price_qtile']:
        df[c] = df[c].astype('category')

    return df

def main():
    train_path = "training_set_VU_DM.csv"
    test_path  = "test_set_VU_DM.csv"

    # 1. 读取原始数据
    print("Load raw data...")
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    # 2. 在原始 DataFrame 上创建一个副本并处理它
    print("Preprocess a modified copy of training data...")
    train_mod = preprocess(train_raw.copy())
    print("Preprocess a modified copy of test data...")
    test_mod  = preprocess(test_raw.copy())

    # 3. 保存 “修改过” 的整表
    train_mod.to_csv("training_set_VU_DM_modified.csv", index=False)
    test_mod.to_csv("test_set_VU_DM_modified.csv",     index=False)

    # 4. 再从 train_mod 划分训练/验证
    train_set, val_set = train_test_split(
        train_mod,
        test_size=0.2,
        stratify=train_mod['booking_bool'],
        random_state=42
    )

    # 5. 保存划分结果
    train_set.to_csv("train_mod_train.csv", index=False)
    val_set.to_csv("train_mod_val.csv",     index=False)

    print("Done. Generated modified datasets without touching the original CSVs.")

if __name__ == "__main__":
    main()
