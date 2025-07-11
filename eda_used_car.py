import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 解决中文乱码和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
file_path = 'data/used_car_train_20200313.csv'
df = pd.read_csv(file_path, delim_whitespace=True)

# 显示所有列
pd.set_option('display.max_columns', None)

# 1. 基本信息
print('数据集基本信息:')
print(df.info())
print('\n数据集前5行:')
print(df.head())

# 2. 缺失值统计
print('\n每列缺失值数量:')
print(df.isnull().sum())

# 3. 描述性统计
print('\n数值型特征描述性统计:')
print(df.describe())

# 4. 目标变量分布
plt.figure(figsize=(8,4))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('价格分布')
plt.xlabel('价格')
plt.ylabel('数量')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.close()

# 5. 类别型特征分布（以brand为例）
plt.figure(figsize=(10,4))
df['brand'].value_counts().head(20).plot(kind='bar')
plt.title('品牌分布（Top20）')
plt.xlabel('品牌')
plt.ylabel('数量')
plt.tight_layout()
plt.savefig('brand_distribution.png')
plt.close()

# 6. 相关性分析
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('数值特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print('\nEDA分析已完成，主要统计图已保存为PNG文件。') 