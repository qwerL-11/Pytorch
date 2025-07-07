import os
import matplotlib.pyplot as plt
# import numpy as np

# 动态读取训练过程保存的数据（假设每行一个数字）
def load_list_from_txt(path):
    with open(path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def plot_loss_acc(train_loss_path, val_loss_path, train_acc_path, val_acc_path):
    train_loss = load_list_from_txt(train_loss_path)
    val_loss = load_list_from_txt(val_loss_path)
    train_acc = load_list_from_txt(train_acc_path)
    val_acc = load_list_from_txt(val_acc_path)

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Train Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 用法示例
if __name__ == '__main__':
    plot_dir = 'log\\vgg16\\7.8\\plot' # 曲线数据保存路径
    plot_loss_acc(os.path.join(plot_dir, 'train_loss.txt'),
                  os.path.join(plot_dir, 'val_loss.txt'),
                  os.path.join(plot_dir, 'train_acc.txt'),
                  os.path.join(plot_dir, 'val_acc.txt'))