import pandas as pd
from scipy.io import loadmat

# 采用驱动端数据
"""
mat文件:97  105  118  130  169  185  197  209  222  234
"""
file_names = ['97.mat', '105.mat', '118.mat', '130.mat', '169.mat', 
              '185.mat', '197.mat', '209.mat', '222.mat', '234.mat']
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']

columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']
DE_12k_1797_data = pd.DataFrame()
for index in range(10):
    # 读取MAT文件
    data = loadmat(f'data\\read_data\\DE_12k_1797\\{file_names[index]}')
    dataList = data[data_columns[index]].reshape(-1)
    DE_12k_1797_data[columns_name[index]] = dataList[:119808] # 确保所有文件都有足够数据
print(DE_12k_1797_data.shape)

DE_12k_1797_data.set_index('de_normal', inplace=True)
DE_12k_1797_data.to_csv('data\\read_data\\DE_12k_1797.csv')
