import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) 读取数据（按需指定 nrows）
train = pd.read_csv('training_set_VU_DM.csv', parse_dates=['date_time'], nrows=200000)

# 2) 数据概览
print("\n== Training head ==")
print(train.head())
print("== Dtypes & Memory usage ==")
train.info(memory_usage='deep')


# 3) 缺失值分析
missing = train.isnull().mean().sort_values(ascending=False)
print("\n== Missing rate of each field ==")
print(missing)

# 4) 数值特征分布（Univariate）
numeric_cols = [
    'price_usd', 'prop_review_score', 'prop_location_score1',
    'visitor_hist_adr_usd', 'orig_destination_distance'
]
for col in numeric_cols:
    plt.figure()
    train[col].dropna().hist(bins=50)
    plt.title(f'{col} distribution')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.show()

# 5) 类别特征分布
cat_cols = ['prop_brand_bool', 'promotion_flag', 'srch_saturday_night_bool']
for col in cat_cols:
    plt.figure()
    train[col].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'{col} Distribution ratio')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.show()

# 6) 目标关联分析（按数值分箱）
bin_cols = ['price_usd', 'prop_review_score', 'visitor_hist_adr_usd']
for col in bin_cols:
    # 等频分箱
    bins = pd.qcut(train[col], 10, duplicates='drop')
    grp = train.groupby(bins)['click_bool'].mean()
    plt.figure()
    grp.plot(marker='o')
    plt.title(f'Click-through rate vs {col} Tenth place')
    plt.xlabel(col)
    plt.ylabel('Click-through rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 7) 时间特征提取 & 分析
train['hour'] = train['date_time'].dt.hour
train['weekday'] = train['date_time'].dt.weekday  # 0=Mon
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train.groupby('hour')['click_bool'].mean().plot(ax=axes[0])
axes[0].set_title('Click rate per hour')
train.groupby('weekday')['click_bool'].mean().plot(ax=axes[1])
axes[1].set_title('Click rate per weekday')
plt.tight_layout()
plt.show()

# 8) 分组比较分析：Family vs Business
# 定义敏感分组
train['group'] = np.where(train['srch_children_count'] > 0, 'family', 'business')
# 样本分布
print("\n== Sample distribution (Family vs Business) ==")
print(train['group'].value_counts(normalize=True))
plt.figure()
train['group'].value_counts().plot(kind='bar')
plt.title('Sample Split: Family vs Business')
plt.xlabel('group')
plt.ylabel('count')
plt.tight_layout()
plt.show()
# CTR 与预订率对比
grp_rate = train.groupby('group')[['click_bool', 'booking_bool']].mean()
print("\n== CTR & Booking Rate by Group ==")
print(grp_rate)
plt.figure()
grp_rate.plot.bar(rot=0)
plt.title('CTR & Booking Rate by Group')
plt.xlabel('group')
plt.ylabel('rate')
plt.tight_layout()
plt.show()
# 关键特征分布对比
for col in ['price_usd', 'prop_review_score']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='group', y=col, data=train)
    plt.title(f'{col} Distribution by Group')
    plt.tight_layout()
    plt.show()

# 8) 排名偏差分析
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train[train['random_bool']==0].groupby('position')['click_bool'].mean().plot(ax=axes[0])
axes[0].set_title('Click rate under normal sorting vs Position')
train[train['random_bool']==1].groupby('position')['click_bool'].mean().plot(ax=axes[1])
axes[1].set_title('Click rate under random sorting vs Position')
plt.tight_layout()
plt.show()

# 9) 数值特征相关性矩阵
corr = train[numeric_cols + ['prop_log_historical_price']].corr()
plt.figure(figsize=(8, 8))
plt.matshow(corr, fignum=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Numerical feature correlation matrix', pad=20)
plt.tight_layout()
plt.show()

# 10) 交互特征示例
train['price_ratio'] = train['price_usd'] / train['visitor_hist_adr_usd']
plt.figure()
train.plot.scatter(x='price_ratio', y='click_bool', alpha=0.3)
plt.title('Click-through rate vs. price of the day / historical average price ratio')
plt.xlabel('price_ratio')
plt.ylabel('click_bool')
plt.tight_layout()
plt.show()
