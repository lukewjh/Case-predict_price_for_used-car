import pandas as pd

print("train_fe.parquet------------------")
# 读取 parquet 文件
df = pd.read_parquet("train_fe.parquet")

# 打印前几行（默认前5行）
# print(df.head())

# 如果想打印全部列，而不是只显示部分列名
pd.set_option('display.max_columns', None)

# 如果还想控制行数，比如显示前10行
print(df.head(10))

print("test_fe.parquet--------------------")
df2 = pd.read_parquet("test_fe.parquet")
# print(df2.head())

pd.set_option('display.max_columns', None)# 如果还想控制行数，比如显示前10行
print(df2.head(10))