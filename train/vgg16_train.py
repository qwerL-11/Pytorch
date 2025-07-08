import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import logging # 日志模块
from datetime import datetime # 时间模块
from tqdm import tqdm # 进度条显示模块

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入VGG16模型
from model.vgg16 import VGG16Model

# python -m train.vgg16_train

def load_data(data_path):
    # 定义数据增强与归一化的转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])  # 归一化处理
    ])
    # 自定义类别顺序和映射
    custom_classes = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 
                      'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']
    custom_class_to_idx = {cls_name: i for i, cls_name in enumerate(custom_classes)}

    class MyImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            # 返回自定义类别顺序和映射
            return custom_classes, custom_class_to_idx

        def __getitem__(self, index):
            # 获取原始数据
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # 返回图片、标签、路径
            return sample, target, path

    train_data = MyImageFolder(os.path.join(data_path, 'train'), transform=transform) # 包装训练集
    val_data = MyImageFolder(os.path.join(data_path, 'val'), transform=transform) # 包装验证集

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # 创建训练集数据加载器
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False) # 创建验证集数据加载器

    return train_loader, val_loader # 返回数据加载器

def train_model(model, train_loader, criterion, optimizer, num_epochs, best_model_path, plot_dir):
    """
    model: 你要训练的神经网络模型（如 VGG16Model 的实例）。
    train_loader: 训练集的数据加载器（DataLoader），用于批量读取训练数据。
    criterion: 损失函数（如 nn.CrossEntropyLoss()），用于计算模型输出与真实标签之间的误差。
    optimizer: 优化器（如 optim.Adam），用于更新模型参数以最小化损失。
    num_epochs: 训练轮数，即整个训练集将被模型完整学习多少次。
    """
    # 初始化最佳验证集准确率
    best_val_acc = 0

    # 训练循环
    for epoch in range(num_epochs):

        # plot:打开文件用于保存loss和acc
        os.makedirs(plot_dir, exist_ok=True)
        train_loss_file = open(os.path.join(plot_dir, 'train_loss.txt'), 'w')
        train_acc_file = open(os.path.join(plot_dir, 'train_acc.txt'), 'w')
        val_loss_file = open(os.path.join(plot_dir, 'val_loss.txt'), 'w')
        val_acc_file = open(os.path.join(plot_dir, 'val_acc.txt'), 'w')

        model.train() # 设置模型为训练模式
        total_loss = 0 # 初始化总损失
        correct = 0 # 初始化正确预测计数
        total = 0 # 初始化周期总样本计数
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'), 1):
            images, labels = images.to(device), labels.to(device) # 将数据移动到设备（GPU或CPU）

            optimizer.zero_grad() # 清零梯度

            outputs = model(images) # 前向传播，获取模型输出
            loss = criterion(outputs, labels) # 计算损失

            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数

            total_loss += loss.item() * images.size(0) # 累加损失
            _, predicted = outputs.max(1) # 获取预测结果
            correct += predicted.eq(labels).sum().item() # 计算正确预测的数量
            total += labels.size(0) # 累加总样本数量

            # 输出当前batch的损失和准确率
            logging.info(f"Epoch{epoch+1}-Batch{batch_idx}  Loss: {loss.item():.4f} | Acc: {correct / total:.4f}")
            tqdm.write(f"Epoch{epoch+1}-Batch{batch_idx}  Loss: {loss.item():.4f} | Acc: {correct / total:.4f}")

        train_acc = correct / total # 计算训练集准确率
        train_loss = total_loss / total # 计算平均损失
        # 输出一个epoch的训练结果
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # plot:保存训练loss和acc到文件
        train_loss_file.write(f"{train_loss}\n")
        train_acc_file.write(f"{train_acc}\n")

        # 验证模型
        val_loss, val_acc = validate_model(model, val_loader, criterion, return_loss=True)

        # plot:保存验证loss和acc到文件
        val_loss_file.write(f"{val_loss}\n")
        val_acc_file.write(f"{val_acc}\n")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path) # 保存模型权重
            logging.info(f"Best Model Updated, Accuracy: {best_val_acc:.4f}\n")
            print(f"Best Model Updated, Accuracy: {best_val_acc:.4f}\n")

         # 关闭文件
        train_loss_file.close()
        train_acc_file.close()
        val_loss_file.close()
        val_acc_file.close()

def validate_model(model, val_loader, criterion, return_loss=True):
    # 验证
    model.eval() # 设置模型为评估模式
    val_dataset_loss = 0 # 初始化验证集损失
    val_correct = 0 # 初始化验证集正确预测计数
    val_total = 0 # 验证集总样本数
    with torch.no_grad(): # 在验证时不需要计算梯度
        for images, labels, _ in val_loader: # 遍历验证集
            images, labels = images.to(device), labels.to(device) # 将数据移动到设备
            outputs = model(images) # 前向传播，获取模型输出
            loss = criterion(outputs, labels) # 计算损失
            val_dataset_loss += loss.item() * images.size(0) # 累计损失
            _, predicted = outputs.max(1) # 获取预测结果
            val_correct += predicted.eq(labels).sum().item() # 累计正确预测的数量
            val_total += labels.size(0) # 累计总样本数量
    val_loss = val_dataset_loss / val_total if val_total > 0 else 0 # 计算验证集平均损失
    val_acc = val_correct / val_total # 计算验证集准确率

    logging.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    if return_loss:
        return val_loss, val_acc
    return val_acc
    

if __name__ == "__main__":

    data_path = 'fault_classification_data'  # 数据集路径
    best_model_path = 'best_model\\best_vgg16_7.8.pth' # 最佳模型保存路径
    log_path = 'log\\vgg16\\7.8\\vgg16_train_7.8.txt' # 日志文件路径
    plot_dir = 'log\\vgg16\\7.8\\plot' # 曲线数据保存路径

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')  # 打印使用的设备
    num_classes = 10 # 类别数量

    model = VGG16Model(num_classes=num_classes).to(device) # 实例化模型并移动到设备

    # 如果存在预训练模型，则加载权重
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded weights from {best_model_path}")

    train_loader, val_loader = load_data(data_path) # 加载数据
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 定义优化器
    num_epochs = 20  # 训练轮数

    # 打印类别映射
    # logging.info(f"类别映射: {train_loader.dataset.class_to_idx}")

    # 查看第一个batch的图片及其标签和路径
    # images, labels, paths = next(iter(train_loader))
    # for path, label in zip(paths, labels):
    #     print(f"{os.path.basename(path)}\t标签: {label.item()}")
    
    # 过拟合小样本测试
    # model.train()
    # data, target, paths = next(iter(train_loader))
    # data, target = data.to(device), target.to(device)
    # print(target.dtype, target.min(), target.max())
    # for i in range(100):
    #     optimizer.zero_grad() # 清零梯度
    #     output = model(data) # 前向传播
    #     loss = criterion(output, target) # 计算损失
    #     loss.backward() # 反向传播
    #     optimizer.step() # 更新参数
    #     print(f'Step {i}, Loss: {loss.item()}')
    
    # 配置日志
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # 确保日志目录存在
    # 清除所有已存在的 handler，确保 basicConfig 生效
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.INFO, encoding='utf-8')
    logging.info(f'训练开始 | 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f"Using device: {device}\n")  # 记录使用的设备

    # 开始训练
    train_model(model, train_loader, criterion, optimizer, num_epochs, best_model_path, plot_dir)

    logging.info(f'训练结束 | 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
