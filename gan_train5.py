import torch
from torch import autograd
#import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
import argparse
import os
#from skimage.io import imread, imsave
from torchvision import utils
import numpy as np
#from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import math

from gan_dataset import joint_image
from gan5 import style_mixer, generator, discriminator

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="Gan_test5", help="Dataset")

parser.add_argument("--iteration", type=int, default=150000, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
parser.add_argument("--r1", type=float, default=10)
parser.add_argument("--d_reg_every", type=int, default=16)
parser.add_argument("--g_reg_every", type=int, default=4)
parser.add_argument("--path_batch_shrink", type=int, default=2)
parser.add_argument("--path_regularize", type=float, default=2)

parser.add_argument("--path1", type=str, default="/data/ymh/drawing/dataset/num_004/image_semi/", help="Main path")
parser.add_argument("--path2", type=str, default="/data/ymh/drawing/dataset/num_003/frame/", help="Main path")
parser.add_argument("--save_path", type=str, default="/data/ymh/drawing/save/num_005/", help="save path")

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

dataset_train = joint_image(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True)# num_workers=8)

##############################################################################################################################
def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    #print(grad.shape)
    path_lengths = torch.sqrt(grad.pow(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

##############################################################################################################################
print("="*40)
print(opt)

styler = style_mixer().to(device)
styler_accum = style_mixer().to(device)
styler_accum.load_state_dict(styler.state_dict())

G = generator().to(device)
G_accum = generator().to(device)
G_accum.load_state_dict(G.state_dict())

D = discriminator().to(device)

g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1)
d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1)

g_param = list(styler.parameters()) + list(G.parameters())

g_optim = optim.Adam(
    g_param,
    lr=opt.lr * g_reg_ratio,
    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
)
d_optim = optim.Adam(
    D.parameters(),
    lr=opt.lr * d_reg_ratio,
    betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
)

mean_path_length = 0
##############################################################################################################################
for iteration in range(opt.iteration):
    print("Iteration:%d" % iteration, end=" ")
    joint, _, image = train_loader.__iter__().next()
    joint = joint.to(device)
    #rand_joint = rand_joint.to(device)
    image = image.to(device)
    
    # ---- Discriminator ---- #
    d_optim.zero_grad()
    styler.eval()
    G.eval()
    D.train()
    
    with torch.no_grad():
        style = styler(joint)
        fake = G(style)
        
    real_pred = D(image, joint)
    fake_pred = D(fake, joint)
    
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    d_loss = real_loss.mean() + fake_loss.mean()
    
    d_loss.backward()
    d_optim.step()
    
    print("D:%f" % d_loss.item(), end=" ")
    
    d_regularize = iteration % opt.d_reg_every == 0
    if d_regularize:
        image.requires_grad = True
        real_pred = D(image, joint)
        r1_loss = d_r1_loss(real_pred, image)
        
        d_optim.zero_grad()
        (opt.r1 / 2 * r1_loss * opt.d_reg_every).backward()
        d_optim.step()

    # ---- Generator ---- #
    g_optim.zero_grad()
    styler.train()
    G.train()
    D.eval()

    style = styler(joint)
        
    fake = G(style)    
    fake_pred = D(fake, joint)
    
    g_loss = F.softplus(-fake_pred).mean()
    
    g_loss.backward()
    g_optim.step()
    
    print("G:%f" % g_loss.item())
    
    g_regularize = iteration % opt.g_reg_every == 0

    if g_regularize:
        g_optim.zero_grad()
        
        path_batch_size = max(1, opt.batch_size // opt.path_batch_shrink)
        style = styler(joint[:path_batch_size])
        
        fake = G(style)

        path_loss, mean_path_length, path_lengths = g_path_regularize(
            fake, style, mean_path_length
            )
        
        weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss
        weighted_path_loss.backward()

        g_optim.step()
    
    accumulate(styler_accum, styler)
    accumulate(G_accum, G)
    
    if iteration % 100 == 0:
        style = styler_accum(joint)
        fake = G_accum(style)
        
        utils.save_image(
            fake[:16],
            os.path.join(save_path, "iter_%07d.png" % iteration),
            nrow=4,
            normalize=True,
            range=(-1, 1),
            )

    if iteration % 10000 == 0:
        torch.save(
            {"G_accum": G_accum.state_dict(),
             "styler": styler.state_dict()
             },
            os.path.join(save_path, "weight_%07d.pt" % iteration)
            )
