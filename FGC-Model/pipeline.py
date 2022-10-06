import numpy as np
import os

import seaborn as sns
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import models
import time
from tqdm.auto import tqdm
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report
import torchvision.models as models
import timm
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate
import albumentations as A

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2286
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 256

# LR = 5e-4
LR = 1e-4
NUM_CLASSES = 5
OPTM_STEP = 1
TTA = 5
EVAL_TTA_EVERY = 5


class CLSDataset(Dataset):

    def __init__(self, txt_file, base_dir='/data/yly/zsg/chryset/final_v4', transforms=None,
                 test=False):
        f = open(txt_file, 'r')
        lines = f.readlines()
        self.base_dir = base_dir
        self.img_names = []
        self.labels = []
        self.transforms = transforms
        for line in lines:
            self.img_names.append(line.split(' ')[0])
            self.labels.append(int(line.split(' ')[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        p = self.img_names[idx]
        label = self.labels[idx]

        p_path = self.base_dir + "/" + p

        image = Image.open(p_path).convert('RGB')
        image = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        #         image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return image, label


train_transforms = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    A.CoarseDropout(p=0.5),
    A.Cutout(p=0.5),
    ToTensorV2(p=1.0),
], p=1.)

valid_transforms = A.Compose([
    #             A.CenterCrop(IMG_SIZE,IMG_SIZE, p=1.),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
], p=1.)


# dataset = CLSDataset('./train.txt', transforms=train_transforms)
# i, l = dataset.__getitem__(89)
# print(i.shape)
# print(l)

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)


class CassvaModel(nn.Module):

    def __init__(self, model_arch='resnet101', n_class=5, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def build_model():
    if TRAINING:
        model = CassvaModel(pretrained=True)
        #         model = VisionTransformer.from_pretrained('ViT-B_16', num_classes=5)

        print('model created with imagenet weights')
    else:
        model = CassvaModel()
        #         model = VisionTransformer.from_name('ViT-B_16', num_classes=5)
        print('model loaded from random weight')

    #     model = nn.DataParallel(model.to(device))
    return model.to(device)


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_metrics(model, dataloader, criterion):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    complete_outputs = []
    complete_labels = []
    # print(len(dataloader))
    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    # print(tk)
    for idx, (images, labels) in enumerate(tk):
        images, labels = images.to(device), labels.to(device).long()
        predicted = model(images)

        loss = criterion(predicted, labels)

        predicted_classes = predicted.argmax(1)
        complete_outputs.extend(predicted_classes.tolist())
        complete_labels.extend(labels.tolist())

        correctly_identified_sum = (predicted_classes == labels).sum().item()
        number_of_images = images.size(0)

        accs.update(correctly_identified_sum / number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    cf_matrix = confusion_matrix(complete_outputs, complete_labels)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="YlGnBu")

    target_names = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
                    "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"]
    print(classification_report(complete_outputs, complete_labels, target_names=target_names))

    return losses.avg, complete_outputs, complete_labels


def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)

    optimizer.zero_grad()
    for idx, (img, labels) in enumerate(tk):

        images, labels = img.to(device), labels.to(device).long()
        predicted = model(images)

        loss = criterion(predicted, labels)

        loss.backward()

        if (idx + 1) % OPTM_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()

        predicted_classes = predicted.argmax(1)
        correctly_identified_sum = (predicted_classes == labels).sum().item()
        number_of_images = images.size(0)

        accs.update(correctly_identified_sum / number_of_images, number_of_images)

        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (images, labels) in enumerate(tk):
            images, labels = images.to(device), labels.to(device).long()
            output_valid = model(images)

            loss = criterion(output_valid, labels)

            losses.update(loss.item(), images.size(0))
            accs.update((output_valid.argmax(1) == labels).sum().item() / images.size(0), images.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg, loss


TRAINING = True
model = build_model()
# model.load_state_dict(torch.load(WEIGHT_FILE, map_location='cuda:0'))
model = model.to(device)

if TRAINING:

    dataset_train = CLSDataset('./train.txt', transforms=train_transforms, test=False)
    dataset_valid = CLSDataset('./test.txt', transforms=valid_transforms, test=True)

    # dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    # dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, \
                                                           patience=5, verbose=True, min_lr=1e-5)

    best_acc = 0
    best_loss = 100

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_model(model, epoch, dataloader_train, criterion, optimizer)
        val_loss, val_acc, loss = test_model(model, dataloader_valid, criterion)

        scheduler.step(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './model/epoch-{}-loss-{:.5}.pth'.format(epoch, best_loss))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './model/epoch-{}-acc-{:.5}.pth'.format(epoch, best_acc))

        print('current_val_acc:', val_acc, 'best_val_acc:', best_acc)

show_metrics(model, dataloader_train, criterion)
