import pandas as pd
import matplotlib.pyplot as plt

# 1) 读入数据（按需指定 nrows）
train = pd.read_csv('training_set_VU_DM.csv', parse_dates=['date_time'], nrows=200000)
test  = pd.read_csv('test_set_VU_DM.csv',  parse_dates=['date_time'], nrows=200000)

# 2) 快速预览
print("=== Training head ===")
print(train.head())
print("\n=== Dtypes & Memory usage ===")
print(train.info(memory_usage='deep'))

# 3) 描述性统计
print("\n--- Numeric features describe() ---")
print(train.describe())

print("\n--- Categorical features value_counts() (sample) ---")
for col in ['prop_starrating','srch_length_of_stay','srch_adults_count']:
    print(f"\n{col} distribution:")
    print(train[col].value_counts().sort_index())

# 4) 缺失值分布
missing = train.isnull().mean().sort_values(ascending=False)
print("\n--- Missing rate per column ---")
print(missing[missing>0])

# 5) 可视化：连续变量直方图 & 箱型图
fig, axes = plt.subplots(2, 2, figsize=(12,8))
train['price_usd'].hist(bins=50, ax=axes[0,0])
axes[0,0].set_title('price_usd Histogram')

train['prop_review_score'].hist(bins=20, ax=axes[0,1])
axes[0,1].set_title('prop_review_score Histogram')

train.boxplot(column='price_usd', ax=axes[1,0])
axes[1,0].set_title('price_usd Boxplot')

train.boxplot(column='visitor_hist_adr_usd', ax=axes[1,1])
axes[1,1].set_title('visitor_hist_adr_usd Boxplot')

plt.tight_layout()
plt.show()

# 6) 相关矩阵热图
import seaborn as sns
num_cols = ['price_usd','prop_log_historical_price','visitor_hist_adr_usd','prop_location_score1']
corr = train[num_cols].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues')
plt.title('Numeric Features Correlation')
plt.show()

# 7) 按标签分组对比（点击率 vs 值）
click_rate = train.groupby('prop_starrating')['click_bool'].mean()
booking_rate = train.groupby('prop_starrating')['booking_bool'].mean()

plt.figure(figsize=(6,4))
click_rate.plot(kind='bar', label='click rate')
booking_rate.plot(kind='bar', label='booking rate', alpha=0.7)
plt.legend()
plt.title('Click/Booking Rate by Star Rating')
plt.show()
