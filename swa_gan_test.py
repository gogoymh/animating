import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.optim.swa_utils import AveragedModel, SWALR

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, default=15000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

device = torch.device("cuda:0")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Configure data loader
os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

steps = 400
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, steps)
swa_generator = AveragedModel(generator)
swa_generator.to(device)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, steps)
swa_discriminator = AveragedModel(discriminator)
swa_discriminator.to(device)

swa_start = 8000
swa_scheduler_G = SWALR(optimizer_G, swa_lr=0.0001)
swa_scheduler_D = SWALR(optimizer_D, swa_lr=0.0001)

for iteration in range(opt.iteration):
    real, _ = dataloader.__iter__().next()
    #for i, (real, _) in enumerate(dataloader):
    real = real.to(device)
    real_label = Tensor(real.size(0), 1).fill_(1.0)
    fake_label = Tensor(real.size(0), 1).fill_(0.0)
    z = Tensor(np.random.normal(0, 1, (real.shape[0], opt.latent_dim)))
        
    ## ---- D ---- ##
    optimizer_D.zero_grad()

    with torch.no_grad():
        if (iteration+1) > swa_start:
            fake = swa_generator(z)
        else:
            fake = generator(z)
        
    real_loss = adversarial_loss(discriminator(real), real_label)
    fake_loss = adversarial_loss(discriminator(fake.detach()), fake_label)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()
        
    ## ---- G ---- ##
    optimizer_G.zero_grad()
        
    fake = generator(z)
    
    if (iteration+1) > swa_start:
        g_loss = adversarial_loss(swa_discriminator(fake), real_label)
    else:
        g_loss = adversarial_loss(discriminator(fake), real_label)

    g_loss.backward()
    optimizer_G.step()

    print(
        "[Iteration %d/%d] [D loss: %f] [G loss: %f]"
        % ((iteration+1), opt.iteration, d_loss.item(), g_loss.item())
        )

    if (iteration+1) % 400 == 0:
        if (iteration+1) > swa_start:
            fake = swa_generator(z)
        else:
            fake = generator(z)
        save_image(fake.data[:25], "images/%d.png" % (iteration+1), nrow=5, normalize=True)
            
    if (iteration+1) > swa_start:
        swa_generator.update_parameters(generator)
        swa_scheduler_G.step()
        swa_discriminator.update_parameters(discriminator)
        swa_scheduler_D.step()
    else:
        scheduler_G.step()
        scheduler_D.step()