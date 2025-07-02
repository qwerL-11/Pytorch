# 时间步长:512 重叠率:0.5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# 切割划分
def split_data_with_overlap(data, time_steps, overlap_ratio=0.5):
    step = int(time_steps * (1 - overlap_ratio))
    samples = []
    for start in range(0, len(data) - time_steps + 1, step):
        samples.append(data[start:start + time_steps])
    return np.array(samples)

# 归一化数据
def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler

# 数据集的制作
def make_datasets(data_file_csv, label_list=None, split_rate=[0.7, 0.2, 0.1], time_steps=512, overlap_ratio=0.5):
    df = pd.read_csv(data_file_csv) # 读取数据
    data = df.values # 转换为numpy数组
    data_norm, scaler = normalize(data) # 归一化数据
    samples = split_data_with_overlap(data_norm, time_steps, overlap_ratio) # 分割数据
    print(f'样本总数: {len(samples)}, 每个样本长度: {samples.shape[1]}') # 打印样本信息
    total = len(samples) # 样本总数
    n_train = int(total * split_rate[0]) # 训练集样本数
    n_val = int(total * split_rate[1]) # 验证集样本数
    train_set = samples[:n_train] # 训练集
    val_set = samples[n_train:n_train + n_val] # 验证集
    test_set = samples[n_train + n_val:] # 测试集
    return train_set, val_set, test_set

# 划分训练集、验证集、测试集
train_set, val_set, test_set = make_datasets(
    'data\\read_data\\DE_12k_1797.csv',
    split_rate=[0.7, 0.2, 0.1],
    time_steps=512,
    overlap_ratio=0.5
)

# 保存数据
dump(train_set, 'train_set.joblib')
dump(val_set, 'val_set.joblib')
dump(test_set, 'test_set.joblib')
