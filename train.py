import torch 
from torch.utils.data import DataLoader # 数据加载模块
from tqdm import tqdm # 进度条显示模块
from torchvision import transforms, datasets # 图像处理模块
from model.cnn import CNN # 导入自定义的CNN模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测是否有GPU可用

# 对图像进行预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像大小调整为224*224
    transforms.ToTensor(), # 将图像转换为Tensor 0-1之间的像素值矩阵
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化处理
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像大小调整为224*224
    transforms.ToTensor(), # 将图像转换为Tensor 0-1之间的像素值矩阵
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化处理
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform) # 训练集
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # 创建训练集数据加载器

test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform) # 测试集
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False) # 创建测试集数据加载器

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0 # 用于累计损失
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU或CPU
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward() # 计算梯度
            optimizer.step() # 更新模型参数
            
            running_loss += loss.item() * inputs.size(0)  # 累计损失
        epoch_loss = running_loss / len(train_loader.dataset) # 计算平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Train_Loss: {epoch_loss:.4f}')

        accuracy = evaluate_model(model, test_loader, criterion) # 在测试集上评估模型
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path) # 保存最佳模型
            print(f'Best Model Updated, Accuracy: {accuracy:.4f}')

def evaluate_model(model, test_loader, criterion):
    model.eval() # 设置模型为评估模式
    test_dataset_loss = 0.0 # 用于累计损失
    correct = 0 # 用于累计正确预测的数量
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到GPU或CPU
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            test_dataset_loss += loss.item() * inputs.size(0) # 累计损失
            preds = outputs.argmax(dim=1) # 获取预测结果
            correct += (preds == labels).sum().item() # 累计正确预测的数量
    avg_loss = test_dataset_loss / len(test_loader.dataset) # 计算平均损失
    accuracy = correct / len(test_loader.dataset) # 计算准确率
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return accuracy

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)  # 保存模型参数到指定路径

if __name__ == '__main__':
    num_epochs = 10  # 设置训练轮数
    learning_rate = 0.001  # 设置学习率
    num_class = 4 # 设置分类数量
    save_path = "best_model/best.pth" # 设置模型保存路径
    model = CNN(num_class).to(device) # 创建CNN模型并移动到GPU或CPU
    print(f'device: {device}') # 打印当前设备信息
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器为Adam

    train_model(model, train_loader, criterion, optimizer, num_epochs)  # 训练模型
    evaluate_model(model, test_loader, criterion)  # 在测试集上评估模型
