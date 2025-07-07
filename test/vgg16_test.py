import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import logging
from datetime import datetime
from tqdm import tqdm

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入VGG16模型
from model.vgg16 import VGG16Model

def load_test_data(data_path):  # 加载测试数据集
    # 定义数据增强与归一化的转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 自定义类别顺序和映射
    custom_classes = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner',
                      'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']
    custom_class_to_idx = {cls_name: i for i, cls_name in enumerate(custom_classes)}

    class MyImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            return custom_classes, custom_class_to_idx

        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target, path  # 返回图片、标签、路径

    test_data = MyImageFolder(os.path.join(data_path, 'test'), transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return test_loader

def test_model(model, test_loader, criterion, device): # 测试模型
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_sum = 0

    # 定义类别名，顺序需与训练时一致
    class_names = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner',
                   'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']

    with torch.no_grad():
        num = 1  # 用于记录图片编号
        for images, labels, paths in tqdm(test_loader, desc='Testing', unit='batch'):
            images, labels = images.to(device), labels.to(device) # 将数据移动到设备上
            outputs = model(images) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            test_loss_sum += loss.item() * images.size(0) # 累计损失
            probs = torch.softmax(outputs, dim=1) # 计算每个类别的概率
            _, predicted = outputs.max(1) # 获取预测结果
            test_correct += predicted.eq(labels).sum().item() # 计算正确预测数量
            test_total += labels.size(0) # 累加总样本数量

            # 打印每张图片的预测类别、真实类别、概率和图片名
            for i in range(images.size(0)):
                pred_idx = predicted[i].item() # 获取预测类别索引
                true_idx = labels[i].item() # 获取真实类别索引
                prob_list = probs[i].cpu().numpy() # 获取当前图片的概率列表
                prob_str = ', '.join([f'{p:.3f}' for p in prob_list]) # 格式化概率字符串
                img_name = os.path.basename(paths[i])  # 获取图片文件名
                logging.info(f'图片({num}): {img_name}[{true_idx}] | 预测: {class_names[pred_idx]}[{pred_idx}] | 概率: [{prob_str}]')
                num += 1

    test_loss = test_loss_sum / test_total if test_total > 0 else 0 # 计算平均损失
    test_acc = test_correct / test_total if test_total > 0 else 0 # 计算准确率
    logging.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}") # 打印测试结果

if __name__ == "__main__":

    data_path = 'fault_classification_data'  # 数据集路径
    best_model_path = 'best_model\\best_vgg16_7.7.pth' # 最佳模型保存路径
    log_path = 'log\\vgg16\\7.7\\vgg16_test_7.7(1).txt' # 日志文件路径

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10

    test_loader = load_test_data(data_path) # 加载测试数据集
    model = VGG16Model(num_classes=num_classes).to(device) # 初始化模型
    model.load_state_dict(torch.load(best_model_path, map_location=device)) # 加载训练好的模型
    criterion = nn.CrossEntropyLoss() # 定义损失函数

    # 配置日志
    # 清除所有已存在的 handler，确保 basicConfig 生效
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.INFO, encoding='utf-8')
    logging.info(f'测试开始 | 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # 测试开始
    test_model(model, test_loader, criterion, device)

    logging.info(f'测试结束 | 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
