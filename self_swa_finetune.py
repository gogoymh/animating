import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os
#from torch.optim.swa_utils import AveragedModel, SWALR

from my_conv_unet import my_unet_full
from semantic_set import multiclass_mask, multiclass_mask_cuda
#import config as cfg

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="character", help="Dataset")

parser.add_argument("--n_class", type=int, default=18, help="Batch size")
parser.add_argument("--n_epoch", type=int, default=300, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

parser.add_argument("--path1", type=str, default="/data/ymh/drawing/dataset/num_006/frame", help="Main path")
parser.add_argument("--path2", type=str, default="/data/ymh/drawing/dataset/num_006/label_mask", help="Main path")
parser.add_argument("--save_path", type=str, default="/data/ymh/drawing/save/num_006", help="save path")

opt = parser.parse_args()

##############################################################################################################################
save_path = os.path.join(opt.save_path, opt.exp_name, opt.dataset)
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.makedirs(save_path)
    print("Save path is created: ",save_path)

##############################################################################################################################
device = torch.device("cuda:0")

dataset_train = multiclass_mask(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True, num_workers=3)
dataset_valid = multiclass_mask_cuda(opt.path1, opt.path2, device)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=opt.batch_size)

##############################################################################################################################

'''
def criterion (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1).mean()
'''
criterion = nn.CrossEntropyLoss()

model = my_unet_full(3,64,opt.n_class).to(device)

model_name = os.path.join("/data/ymh/drawing/save/num_006/Drawing/character", "self_sup.pth")
checkpoint = torch.load(model_name)
'''
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    if k == "n_averaged":
        pass
    else:
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
'''
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

model.eval()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
#swa_model = AveragedModel(model)
#swa_start = 200
#swa_scheduler = SWALR(optimizer, swa_lr=0.001)

for epoch in range(opt.n_epoch):
    running_loss=0
    model.train()
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.long().to(device)
            
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    '''
    if (epoch+1) > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()
    '''  
    running_loss /= len(train_loader)    
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))

#torch.optim.swa_utils.update_bn(valid_loader, swa_model)

model_name = os.path.join(save_path, "swa_finetune.pth")
torch.save({'model_state_dict': model.state_dict()}, model_name)
    

