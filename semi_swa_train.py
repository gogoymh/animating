import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os
from skimage.io import imread, imsave
from torchvision import transforms
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from my_net import Our_Unet_singlegpu_Full
from semantic_set import multiclass_onehot, multiclass_onehot_cuda
import some_loss

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="joint", help="Dataset")
parser.add_argument("--init_test", type=int, default=1, help="Initial test index")
parser.add_argument("--repeat", type=int, default=10, help="How many repeat")

parser.add_argument("--n_epoch", type=int, default=50, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

parser.add_argument("--path1", type=str, default="/home/compu/ymh/drawing/dataset/num_003/frame/", help="Main path")
parser.add_argument("--path2", type=str, default="/home/compu/ymh/drawing/dataset/num_004/onehot_pseudo/", help="Main path")
parser.add_argument("--save_path", type=str, default="/home/compu/ymh/drawing/save/num_004/", help="save path")

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

dataset_train = multiclass_onehot(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

dataset_valid = multiclass_onehot_cuda(opt.path1, opt.path2, device)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=opt.batch_size)

##############################################################################################################################
criterion = some_loss.softXEnt()

inference_output = nn.Softmax2d().to(device)

print("="*40)
print(opt)

for test_idx in range(opt.init_test,(opt.init_test + opt.repeat)):
    print("="*40, end=" ")
    print("Test #%d" % test_idx, end=" ")
    print("="*40)
    
    model = Our_Unet_singlegpu_Full(8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    steps = 10 #dataset_train.len // opt.batch_size + 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    
    swa_model = AveragedModel(model)
    swa_start = 20
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    
    best_f1 = 0
    for epoch in range(opt.n_epoch):
        running_loss=0
        model.train()
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #print(loss.item())
            
            scheduler.step()
            #print(scheduler.get_lr())
            
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        running_loss /= len(train_loader)
        
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))
    
    print("="*40, end=" ")
    print("BN update", end=" ")
    print("="*40)
    #swa_model.cpu()
    torch.optim.swa_utils.update_bn(valid_loader, swa_model)
    
    model_name = os.path.join(save_path, "semi_swa_%02d.pth" % test_idx)
    torch.save({'model_state_dict': swa_model.state_dict()}, model_name)
    
    print("Done")
    
    print("="*40, end=" ")
    print("inference", end=" ")
    print("="*40)
    swa_model.to(device)
    swa_model.eval()
    for index in tqdm(range(dataset_valid.len)):
        img, onehot = dataset_valid.__getitem__(index)
        img = img.unsqueeze(0)#.float().to(device)
        
        output = inference_output(swa_model(img))
        output = output.squeeze().detach().clone()
        
        onehot = onehot.squeeze()#.to(device)
        onehot = 0.9 * onehot + 0.1 * output
        onehot = onehot.cpu().numpy()
        
        np.save(os.path.join(opt.path2, "onehot_%07d" % index), onehot)
