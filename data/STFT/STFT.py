import numpy as np
import pandas as pd
from joblib import load
from scipy.signal import stft
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

"""
参数选择
取CSV表中的四列数据（假设列名如下，需与实际表头一致）
# data_list1 = df['de_normal'].values[:1024]
# data_list2 = df['de_21_inner'].values[:1024]
# data_list3 = df['de_21_ball'].values[:1024]
# data_list4 = df['de_21_outer'].values[:1024]
"""
def stft_params():
    # 导入CSV数据
    df = pd.read_csv('data\\read_data\\DE_12k_1797.csv')

    # 四列数据及其标签
    col_names = ['de_normal', 'de_21_inner', 'de_21_ball', 'de_21_outer']
    scale_list = [32, 64, 128, 256]

    plt.figure(figsize=(26, 26), dpi=100)

    for row, col_name in enumerate(col_names):
        data_list = df[col_name].values[:1024]
        for col, window_size in enumerate(scale_list):
            overlap = 0.5
            overlap_samples = int(window_size * overlap)
            frequencies, times, magnitude = stft(data_list, nperseg=window_size, noverlap=overlap_samples)
            plt.subplot(4, 4, row * 4 + col + 1)
            plt.pcolormesh(times, frequencies, np.abs(magnitude), shading='gouraud')
            plt.title(f'{col_name} - 重叠 {window_size*overlap}')
            # plt.xlabel('Time')
            # plt.ylabel('Frequency')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # 调整布局以适应标题
    plt.subplots_adjust(hspace=0.1,wspace=0.1)   # 调整子图间距
    plt.savefig('data\\STFT\\stft_result.png', bbox_inches='tight', dpi=300)  # 保存图片
    plt.show()


# 导入数据
# train_set = load('data\\make_datasets\\train_set.joblib')
# val_set = load('data\\make_datasets\\val_set.joblib')
# test_set = load('data\\make_datasets\\test_set.joblib')

if __name__ == '__main__':

    # 调用STFT参数选择函数
    stft_params()