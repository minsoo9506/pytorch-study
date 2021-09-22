import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List

def Anomaly_score(x, fake_img, D, Lambda=0.1):
    residual_loss = torch.abs(x - fake_img)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x)
    _, G_feature = D(fake_img)

    discrimination_loss = torch.abs(x_feature - G_feature)
    discrimination_loss = discrimination_loss.view(
        discrimination_loss.size()[0], -1
    )
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss
    total_loss = torch.sum(loss)
    return total_loss, loss, residual_loss

def make_datapath_list() -> List:
    train_img_list = []
    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)
        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)
    return train_img_list

def make_test_datapath_list() -> List:
    train_img_list = []
    for img_idx in range(5):
        img_path = './data/test/img_7_' + str(img_idx) + '.jpg'
        train_img_list.append(img_path)
        img_path = './data/test/img_8_' + str(img_idx) + '.jpg'
        train_img_list.append(img_path)      
        img_path = './data/test/img_2_' + str(img_idx) + '.jpg'
        train_img_list.append(img_path)
    return train_img_list

class GAN_Img_Dataset(Dataset):
    def __init__(self, file_list, mean, std):
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path) 
        img_transformed = self.transform(img)
        return img_transformed  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    