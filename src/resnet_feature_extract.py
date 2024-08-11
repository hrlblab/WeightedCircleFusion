# import torch
# from torchvision import transforms
# from torchvision.models import resnet50
# from PIL import Image
#
# # 加载预训练的ResNet18模型
# model = resnet50(pretrained=True)
#
# # 将模型设置为评估模式
# model.eval()
#
# # 修改模型以用于特征提取
# # 移除最后的全连接层，获取前一层的输出作为特征
# feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#
# # 图像预处理
# # ResNet期望的输入是224x224的图像，且进行了特定的标准化
# preprocess = transforms.Compose([
#     transforms.Resize(256),           # 先缩放图像
#     transforms.CenterCrop(224),       # 再中心裁剪至224x224
#     transforms.ToTensor(),            # 将PIL图像转换为Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # 加载你的图像
# image = Image.open("/data/CircleNet/data/WSI_raw_data/14-hypertension_mouse/14_hy_patch/204549/204549_3239_9118_1.0||0.66.png").convert("RGB")  # 确保图像为RGB格式
#
# # 预处理图像
# input_tensor = preprocess(image)
#
# # 添加batch维度，因为模型期望的输入是一个batch
# input_batch = input_tensor.unsqueeze(0)
#
# # 确保使用的是不进行梯度计算（用于推理）
# with torch.no_grad():
#     # 获取特征
#     features = feature_extractor(input_batch)
#
# # 因为特征还有一个无用的batch维度和一个维度为1的维度，我们可以使用squeeze()方法去除它们
# features = features.squeeze().numpy()
# print(features)


import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import numpy as np

# 加载预训练的ResNet50模型
model = resnet50(pretrained=True)

# 将模型设置为评估模式
model.eval()

# 修改模型以用于特征提取
# 移除最后的全连接层，获取前一层的输出作为特征
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# 图像预处理
# ResNet期望的输入是224x224的图像，且进行了特定的标准化
preprocess = transforms.Compose([
    transforms.Resize(256),           # 先缩放图像
    transforms.CenterCrop(224),       # 再中心裁剪至224x224
    transforms.ToTensor(),            # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 图像文件夹路径
image_folder = "/data/CircleNet/data/KPMP/KPMP_Patch_1400*1400"

# 用于存储所有图片特征的列表
all_features = []

# 遍历图像文件夹及其子文件夹中的所有图片
for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if filename.endswith(".png"):
            # 加载图像
            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert("RGB")

            # 预处理图像
            input_tensor = preprocess(image)

            # 添加batch维度，因为模型期望的输入是一个batch
            input_batch = input_tensor.unsqueeze(0)

            # 确保使用的是不进行梯度计算（用于推理）
            with torch.no_grad():
                # 获取特征
                features = feature_extractor(input_batch)

            # 因为特征还有一个无用的batch维度和一个维度为1的维度，我们可以使用squeeze()方法去除它们
            features = features.squeeze().numpy()

            # 存储特征
            all_features.append(features)

# 将所有特征转换为numpy数组
all_features = np.array(all_features)

# 打印所有特征
print(all_features)


from sklearn.cluster import KMeans

# 聚类数量
num_clusters = 5  # 你可以根据需要调整聚类数量

# 使用KMeans算法对特征进行聚类
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_features)

# 获取聚类结果
labels = kmeans.labels_

# 打印每个样本所属的类别
print(labels)
