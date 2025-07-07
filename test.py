import torch
from model.cnn_template import CNN


x = torch.randn(32, 3, 224, 224) # 模拟输入数据，batch_size为32，通道数为3，图片大小为224*224
model = CNN(4) # 创建CNN模型，num_classes为分类数量
output = model(x) # 前向传播
print(output.shape) # 输出结果的形状，应该为(32, 4)

import torch
print(torch.cuda.is_available())
