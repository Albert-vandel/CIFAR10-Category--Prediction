
# CIFAR10图像分类项目

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

使用LeNet和ResNet34模型实现CIFAR10数据集的图像分类，包含完整训练流程、可视化分析和模型部署能力。

## 目录
- [项目亮点](#项目亮点)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [模型架构](#模型架构)
- [训练流程](#训练流程)
- [结果展示](#结果展示)
- [模型导出](#模型导出)

## 项目亮点
- 🚀 实现经典LeNet与深度ResNet34模型的对比实验
- 📊 完整可视化支持：损失曲线、混淆矩阵、预测热力图
- 🛠️ 支持ONNX模型导出与GPU加速训练
- 🔍 详细的超参数配置说明与可复现性保证

## 环境要求
```bash
# 基础依赖
pip install torch==1.13.1 torchvision==0.14.1
# 可视化工具
pip install matplotlib seaborn
# 模型导出
pip install onnx
```

## 快速开始
### 训练模型
```python
# 训练LeNet
python train.py --model lenet --epochs 15 --batch_size 128

# 训练ResNet34 
python train.py --model resnet34 --epochs 30 --batch_size 64
```

### 测试与可视化
```python
# 生成混淆矩阵
python visualize.py --model lenet --plot confusion_matrix

# 导出训练曲线
python visualize.py --plot loss_curve
```

## 数据集
使用CIFAR10标准数据集：
- 50,000训练图像 + 10,000测试图像
- 32x32 RGB图像，10个类别（飞机、汽车等）
- 预处理流程：
  ```python
  transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  ```

## 模型架构
### LeNet改进版
```text
CNN层:
Conv1 (3→6通道, 5x5核) → ReLU → MaxPool2d
Conv2 (6→16通道, 5x5核) → ReLU → MaxPool2d

全连接层:
16*5*5 → 120 → 84 → 10
```

### ResNet34适配
```text
输入适配层:
Conv2d(3,64,kernel_size=3,stride=1,padding=1)
移除原始maxpool层

输出层:
全连接层(512→10)
```

## 训练流程
```python
# 核心配置
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# 训练监控指标
Epoch 1/15 | Train Loss: 1.5123 | Val Acc: 45.67%
...
Epoch 15/15 | Train Loss: 0.2174 | Val Acc: 76.89%
```

## 结果展示
### 性能对比
| 模型    | 参数量 | 测试准确率 | 训练时间(epoch) |
|---------|--------|------------|-----------------|
| LeNet   | 60K    | 67.2%      | 2min            |
| ResNet34| 21M    | 82.1%      | 8min            |

![验证准确率曲线](docs/accuracy_curve.png)
![混淆矩阵](docs/confusion_matrix.png)

## 模型导出
导出为ONNX格式：
```python
python export_onnx.py --model resnet34 --input_size 3 32 32
```
支持特性：
- 动态batch尺寸
- 算子版本兼容性(Opset 12)
- 包含模型元数据
