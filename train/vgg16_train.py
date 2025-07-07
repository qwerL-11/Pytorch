import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.vgg16 import VGG16Model
from tqdm import tqdm # 进度条显示模块

# python -m train.vgg16_train

def load_data(data_path):
    # 定义数据增强与归一化的转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小
        transforms.ToTensor(), # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])  # 归一化处理
    ])

    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform) # 包装训练集
    val_data = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform) # 包装验证集

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # 创建训练集数据加载器
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False) # 创建验证集数据加载器

    return train_loader, val_loader # 返回数据加载器

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    model: 你要训练的神经网络模型（如 VGG16Model 的实例）。
    train_loader: 训练集的数据加载器（DataLoader），用于批量读取训练数据。
    criterion: 损失函数（如 nn.CrossEntropyLoss()），用于计算模型输出与真实标签之间的误差。
    optimizer: 优化器（如 optim.Adam），用于更新模型参数以最小化损失。
    num_epochs: 训练轮数，即整个训练集将被模型完整学习多少次。
    """
    # 训练与验证
    best_val_acc = 0 # 初始化最佳验证集准确率
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        total_loss = 0 # 初始化总损失
        correct = 0 # 初始化正确预测计数
        total = 0 # 初始化周期总样本计数
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
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
        train_acc = correct / total # 计算训练集准确率
        train_loss = total_loss / total # 计算平均损失
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # 验证模型
        val_acc = validate_model(model, val_loader, criterion)
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model\\best_vgg16_model.pth')
            print(f'Best Model Updated, Accuracy: {best_val_acc:.4f}')
def validate_model(model, val_loader, criterion):
    # 验证
    model.eval() # 设置模型为评估模式
    val_dataset_loss = 0 # 初始化验证集损失
    val_correct = 0 # 初始化验证集正确预测计数
    val_total = 0 # 验证集总样本数
    with torch.no_grad(): # 在验证时不需要计算梯度
        for images, labels in val_loader: # 遍历验证集
            images, labels = images.to(device), labels.to(device) # 将数据移动到设备
            outputs = model(images) # 前向传播，获取模型输出
            loss = criterion(outputs, labels) # 计算损失
            val_dataset_loss += loss.item() * images.size(0) # 累计损失
            _, predicted = outputs.max(1) # 获取预测结果
            val_correct += predicted.eq(labels).sum().item() # 累计正确预测的数量
            val_total += labels.size(0) # 累计总样本数量
    val_loss = val_dataset_loss / val_total if val_total > 0 else 0 # 计算验证集平均损失
    val_acc = val_correct / val_total # 计算验证集准确率

    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return val_acc
    

if __name__ == "__main__":

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')  # 打印使用的设备
    num_classes = 10
    # 定义VGG16网络结构
    vgg16_conv_arch = [
        (2, 64),
        (2, 128),
        (3, 256),
        (3, 512),
        (3, 512)
    ]

    data_path = 'fault_classification_data'  # 数据集路径

    model = VGG16Model(num_classes=num_classes, conv_arch=vgg16_conv_arch).to(device) # 实例化模型并移动到设备
    train_loader, val_loader = load_data(data_path) # 加载数据

    # 查看batch结构
    images, labels = next(iter(train_loader))
    print("训练数据：")
    print("图像尺寸:", images.shape)  # 应该为 [batch, 3, 224, 224]
    print("标签:", labels)

    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 定义优化器
    num_epochs = 50  # 训练轮数

    # 过拟合小样本测试
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    for i in range(100):
        optimizer.zero_grad() # 清零梯度
        output = model(data) # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        print(f'Step {i}, Loss: {loss.item()}')

    # 判断是否有GPU可用
    # for images, labels in train_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     print(images.device, next(model.parameters()).device)
    #     break

    train_model(model, train_loader, criterion, optimizer, num_epochs)  # 开始
