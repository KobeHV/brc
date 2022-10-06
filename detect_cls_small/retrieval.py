import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from resnet import resnet50
from model import inference
from PIL import Image

BASE_PATH = "./retrieval_db/"

state = 'pre-trained_model/dl_model/epoch-5-acc-0.99501.pth'
model = resnet50(pretrained=False)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 5)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.eval()
model.load_state_dict(torch.load(state, map_location=device), strict=True)
model = model.to(device)
#
img_list = os.listdir(BASE_PATH)
print(len(img_list), img_list)
print("features:\n")
features_list = []
for img_name in img_list:
    img_pth = os.path.join(BASE_PATH, img_name)
    print(img_pth)
    image = Image.open(img_pth).convert('RGB')
    image = inference(image)
    image = image.to(device)
    feature, _ = model(image)
    feature = feature.cpu().detach().numpy()
    del image
    # print("feature:", feature.shape)
    features_list.append(feature)
    del feature
print(len(features_list))
# python保存.mat文件
save_fn = './restore/img_list.mat'
sio.savemat(save_fn, {'img_list': img_list})  # 和上面的一样，存在了array变量的第一行
save_fn = './restore/features_list.mat'
sio.savemat(save_fn, {'features_list': features_list})  # 和上面的一样，存在了array变量的第一行
#
# # python加载.mat文件
# load_fn = './restore/img_list.mat'
# load_data = sio.loadmat(load_fn)
# x = load_data["img_list"]
# print(type(x), x)
# load_fn = './restore/features_list.mat'
# load_data = sio.loadmat(load_fn)
# x = load_data["features_list"]
# print(type(x), x[1].shape)
