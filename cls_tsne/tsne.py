import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from resnet import resnet50
from model import inference
from PIL import Image
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
import matplotlib.pyplot as plt

BASE_PATH = "D:/BRC-Project/FGC/datasets/tsne/"

state = 'pre-trained_model/dl_model/epoch-28-loss-0.0045224.pth'
model = resnet50(pretrained=False)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.load_state_dict(torch.load(state, map_location=device), strict=True)
model = model.to(device)

# img_list = os.listdir(BASE_PATH)
# # print(len(img_list), img_list)
# print("features:\n")
# label_list = []
# features_list = []
#
# for img_name in img_list:
#     img_pth = os.path.join(BASE_PATH, img_name)
#     # print(img_pth)
#
#     image = inference(img_pth)
#     image = image.to(device)
#     _, feature = model(image)
#
#     label = nn.functional.softmax(feature)
#     # score = format(label.cpu().max().detach().numpy().item() * 100, '.2f')
#     label = label.cpu().argmax(1).detach().numpy()[0]
#
#     feature = feature.cpu().detach().numpy()
#     # print("x:", label, feature.shape, feature)
#     del image
#     # print("feature:", feature.shape)
#     label_list.append(label)
#     features_list.append(feature)
#     del feature
#
# print(len(label_list))
# print(len(features_list))

# python保存.mat文件
# save_fn = './restore/tsne_label_list.mat'
# sio.savemat(save_fn, {'tsne_label_list': label_list})  # 和上面的一样，存在了array变量的第一行
# save_fn = './restore/tsne_feature_list.mat'
# sio.savemat(save_fn, {'tsne_feature_list': features_list})  # 和上面的一样，存在了array变量的第一行

load_label = 'restore/label_list.mat'
load_label = sio.loadmat(load_label)
label_list = load_label["tsne_label_list"]
label = np.array(label_list)
label = np.squeeze(label, axis=0)

load_feature = 'restore/features_list.mat'
load_feature = sio.loadmat(load_feature)
feature_list = load_feature["tsne_feature_list"]
feature = np.array(feature_list)
feature = np.squeeze(feature, axis=1)
# print()

im = feature
la = label
# im, la = im.numpy(), la.numpy()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(im)

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(la[i]), color=plt.cm.Set1(la[i]),
             fontdict={'weight': 'bold', 'size': 20})
plt.xticks([])
plt.yticks([])
plt.show()
plt.save("restore/tsne.png", plt)

# # python加载.mat文件
# load_fn = './restore/img_list.mat'
# load_data = sio.loadmat(load_fn)
# x = load_data["img_list"]
# print(type(x), x)
# load_fn = './restore/features_list.mat'
# load_data = sio.loadmat(load_fn)
# x = load_data["features_list"]
# print(type(x), x[1].shape)
