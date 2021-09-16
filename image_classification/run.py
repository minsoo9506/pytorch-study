from LoadData import HymenopteraDataset
from train import train_model
from utils import ImageTransform, make_datapath_list

from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim
import torch

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_list = make_datapath_list(mode='train')
val_list = make_datapath_list(mode='val')

train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(resize, mean, std), mode='train'
)
val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(resize, mean, std), mode='val'
)

batch_size = 32

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

dataloaders_dict = {'train' : train_dataloader, 'val' : val_dataloader}

use_pretrained = True
net = vgg16(pretrained=use_pretrained)
net.classifier[6] == nn.Linear(in_features=4096, out_features=2)

criterion = nn.CrossEntropyLoss()

params_to_update = []
update_param_names = ['classifier.6.weight', 'classifier.6.bias']
for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

if  torch.cuda.is_available():
    device = torch.device('cuda:0')

train_model(net, dataloaders_dict, criterion, optimizer, 10, 'train', device)

# save_path = './weights_vgg16.pth'
# torch.save(net.state_dict(), save_path))

# load_path = './weights_vgg16.pth'
# load_weights = torch.load(load_path)
# net.load_state_dict(load_weights)