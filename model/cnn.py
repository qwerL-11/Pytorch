import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes): # num_classes为分类数量
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长, padding 16*224*224
            nn.ReLU(), # 卷积之后接上激活函数ReLU，增加非线性特征
            nn.MaxPool2d(2, 2),      # 最大池化层，池化窗口大小为2x2，步长为2 16*112*112
            nn.Conv2d(16, 32, 3, 1, 1), # 第二层卷积
            nn.ReLU(), # 第二层卷积之后接上激活函数ReLU
            nn.MaxPool2d(2, 2) # 第二层卷积之后接上最大池化层 32*56*56
        )
        # 全连接层 做分类
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128), # 全连接层，输入通道数*图片大小，输出通道数 512
            nn.ReLU(), # 全连接层之后接上激活函数ReLU
            nn.Linear(128, num_classes) # 全连接层，输入通道数，输出通道数 num_classeswe 为分类数量
        )
    def forward(self, x):
        # 前向传播
        x = self.features(x) # 先将图像通过特征提取层
        x = x.view(x.size(0), -1) # 展平操作，将多维张量展平为一维张量，x.size(0)为batch_size
        x = self.classifier(x) # 再通过分类层
        return x # 返回分类结果
