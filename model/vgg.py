import torch
import torch.nn as nn

class VGG16Model(nn.Module):
    def __init__(self, num_classes, conv_arch, input_channels=3):
        super(VGG16Model, self).__init__() # 初始化父类
        self.num_classes = num_classes # 分类数量
        self.conv_arch = conv_arch # 卷积层结构 [(卷积层数, 输出通道数), ...]
        self.input_channels = input_channels # 输入通道数，默认为3（RGB图像）
        self.features = self.make_layers() # 创建卷积层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 自适应平均池化，将输出大小调整为7x7
        self.classifier = nn.Sequential( # 全连接层结构
            nn.Flatten(), # 将张量展平
            nn.Linear(512 * 7 * 7, 4096), # 全连接层，输入通道数*图片大小，输出通道数
            nn.ReLU(), # 全连接层之后接上激活函数ReLU
            nn.Dropout(0.5), # 添加Dropout层
            nn.Linear(4096, 4096), # 全连接层，输入通道数，输出通道数 num_classes为分类数量
            nn.ReLU(), # 全连接层之后接上激活函数ReLU
            nn.Dropout(0.5), # 添加Dropout层
            nn.Linear(4096, num_classes), # 最后一层全连接层，输出通道数为分类数量
            nn.Softmax(dim=1) # Softmax激活函数，输出概率分布
        )

    def make_layers(self): # 创建卷积层结构
        layers = [] # 初始化卷积层列表
        in_channels = self.input_channels # 输入通道数
        # 遍历卷积层结构，创建对应的卷积层和池化层
        # conv_arch 是一个列表，包含每个卷积块的层数和输出
        # num_convs, out_channels = conv_arch[i] # 获取每个卷积块的层数和输出通道数
        for num_convs, out_channels in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 添加最大池化层
        return nn.Sequential(*layers) # 将所有层组合成一个顺序容器

    def forward(self, x):
        x = self.features(x) # 先通过卷积层
        x = self.avgpool(x) # 再通过平均池化层
        x = self.classifier(x) # 最后通过全连接层
        return x

# 用法示例
if __name__ == '__main__':

    # VGG16 的卷积结构
    vgg16_conv_arch = [
        (2, 64),
        (2, 128),
        (3, 256),
        (3, 512),
        (3, 512)
    ]

    # 示例：假设图片是3通道，分类数为10
    model = VGG16Model(num_classes=10, conv_arch=vgg16_conv_arch, input_channels=3) 
    x = torch.randn(2, 3, 224, 224)  # batch_size=2, 3通道, 224x224
    out = model(x)
    print(out.shape)  # 应为 [2, 10]
    