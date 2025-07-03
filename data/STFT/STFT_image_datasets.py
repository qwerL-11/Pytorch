import os
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# 加载数据
train_set = load('data\\make_datasets\\train_set.joblib')
val_set = load('data\\make_datasets\\val_set.joblib')
test_set = load('data\\make_datasets\\test_set.joblib')
data_set = [train_set, val_set, test_set]

# 数据集保存路径
train_path = 'fault_classification_data/train/'
val_path = 'fault_classification_data/val/'
test_path = 'fault_classification_data/test/'
path_list = [train_path, val_path, test_path]

def makeTimeFrequencyImage(data, img_path, channel_names=None, img_size=(224, 224)):
    os.makedirs(img_path, exist_ok=True)
    for i, sample in enumerate(data):
        if sample.ndim == 2:  # 多通道
            for ch in range(sample.shape[1]):
                signal = sample[:, ch]
                overlap = 0.5  # 重叠率
                window_size = 128  # 窗口大小
                noverlap = int(window_size * overlap)  # 重叠点数
                f, t, Zxx = stft(signal, nperseg=window_size, noverlap=noverlap)
                # plt.figure(figsize=(3, 3))
                plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
                plt.axis('off')
                plt.tight_layout(pad=0)
                # 用通道名命名
                ch_name = channel_names[ch] if channel_names else f'ch{ch}'
                plt.savefig(os.path.join(img_path, f'{ch_name}_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
        else:  # 单通道
            signal = sample
            f, t, Zxx = stft(signal, nperseg=128)
            plt.figure(figsize=(3, 3))
            plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            plt.axis('off')
            plt.tight_layout(pad=0)
            ch_name = channel_names[0] if channel_names else 'ch0'
            plt.savefig(os.path.join(img_path, f'{i}_{ch_name}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

def GenerateImageDataset(path_list, data_set, channel_names=None):
    for path, data in zip(path_list, data_set):
        makeTimeFrequencyImage(data, path, channel_names=channel_names)

# 假设你的通道名如下（根据实际情况填写）
channel_names = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']

# 生成图片数据集
GenerateImageDataset(path_list, data_set, channel_names=channel_names)
