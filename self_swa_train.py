import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os
#from torch.optim.swa_utils import AveragedModel, SWALR

from my_conv_unet import my_unet, Unet_head
from semantic_set import self_set, ordinary_set
import some_loss

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="character", help="Dataset")

parser.add_argument("--n_epoch", type=int, default=1500, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

#parser.add_argument("--path1", type=str, default="/home/compu/ymh/drawing/dataset/num_001/frames", help="Main path")
#parser.add_argument("--save_path", type=str, default="/home/compu/ymh/drawing/save/num_001", help="save path")

parser.add_argument("--path1", type=str, default="/data/ymh/drawing/dataset/num_006/frame/", help="Main path")
parser.add_argument("--save_path", type=str, default="/data/ymh/drawing/save/num_006/", help="save path")

opt = parser.parse_args()

##############################################################################################################################
save_path = os.path.join(opt.save_path, opt.exp_name, opt.dataset)
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.makedirs(save_path)
    print("Save path is created: ",save_path)

device = torch.device("cuda:0")
##############################################################################################################################
dataset_train = self_set(opt.path1)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True)#, num_workers=12)

dataset_valid = ordinary_set(opt.path1, device)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=opt.batch_size)

num_iter = 250 // opt.batch_size + 1

print(num_iter)
##############################################################################################################################
model = my_unet(3,64,20).to(device)
encoding = Unet_head().to(device)

params = list(model.parameters()) + list(encoding.parameters())
optimizer = optim.Adam(params, lr=opt.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

'''
swa_model = AveragedModel(model)
swa_model.to(device)
swa_start = 200
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)"
'''

contrastive_loss = some_loss.NTXentLoss(device, opt.batch_size)

#save_list_1 = [i for i in range(10,110,10)]
#save_list_2 = [i for i in range(100,1550,100)]
#save_list = save_list_1 + save_list_2

print("="*100)
print(opt)
print("="*100)
for epoch in range(opt.n_epoch):
    running_loss = 0
    for i in range(num_iter):
        x1, x2 = train_loader.__iter__().next()
        optimizer.zero_grad()
        
        x1 = x1.float().to(device)
        x2 = x2.float().to(device)
        
        out1 = encoding(model(x1))
        out2 = encoding(model(x2))
        
        loss = contrastive_loss(out1, out2)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        #print(loss.item())
    
    '''
    if (epoch+1) > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()
    '''
    scheduler.step()
    
    running_loss /= num_iter
    print("[Epoch:%d] [loss:%f]" % (epoch+1, running_loss))

    model_name = os.path.join(save_path, "self_sup.pth")
    torch.save({'model_state_dict': model.state_dict()}, model_name)

'''
torch.optim.swa_utils.update_bn(valid_loader, swa_model)
model_name = os.path.join(save_path, "self_sup.pth")
torch.save({'model_state_dict': swa_model.state_dict()}, model_name)
'''  
  
print("saved.")
    
'''
    if (epoch+1) in save_list:
        model_name = os.path.join(save_path, "self_sup_%04d.pth" % (epoch+1))
        torch.save({'model_state_dict': model.state_dict()}, model_name)
        print("Model saved.")
    else:
        print("")
'''