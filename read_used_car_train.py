import pandas as pd

# 读取CSV文件，指定分隔符为任意空白字符（包括空格和Tab）
# file_path = 'data/used_car_train_20200313.csv'
file_path = 'data/used_car_testB_20200421.csv'
df = pd.read_csv(file_path, delim_whitespace=True)

# 设置显示所有列
pd.set_option('display.max_columns', None)

# 显示前5行
print(df.head())