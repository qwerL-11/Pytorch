import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.vgg16 import VGG16Model

def load_test_data(data_path): # 加载测试数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_data = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return test_loader

def test_model(model, test_loader, criterion, device): # 测试模型
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_sum = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # 将数据移动到设备上
            outputs = model(images) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            test_loss_sum += loss.item() * images.size(0) # 累计损失
            _, predicted = outputs.max(1) # 获取预测结果
            test_correct += predicted.eq(labels).sum().item() # 计算正确预测数量
            test_total += labels.size(0) # 累加总样本数量
    test_loss = test_loss_sum / test_total if test_total > 0 else 0 # 计算平均损失
    test_acc = test_correct / test_total if test_total > 0 else 0 # 计算准确率
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    vgg16_conv_arch = [
        (2, 64),
        (2, 128),
        (3, 256),
        (3, 512),
        (3, 512)
    ]

    data_path = 'fault_classification_data' # 数据集路径
    test_loader = load_test_data(data_path) # 加载测试数据集
    model = VGG16Model(num_classes=num_classes, conv_arch=vgg16_conv_arch).to(device) # 初始化模型
    model.load_state_dict(torch.load('vgg16_model.pth', map_location=device)) # 加载训练好的模型
    criterion = nn.CrossEntropyLoss() # 定义损失函数

    test_model(model, test_loader, criterion, device) # 测试模型
