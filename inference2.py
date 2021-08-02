import os
import torch
from torchvision import transforms
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

from my_conv_unet import my_unet_full
from torch.optim.swa_utils import AveragedModel

device = torch.device("cuda:0")
model = Our_Unet_singlegpu_Full(8).to(device)
swa_model = AveragedModel(model)

save_path = "/home/compu/ymh/drawing/save/num_004/Drawing/joint"

index = 2
model_name = os.path.join(save_path, "semi_swa_%02d.pth" % index)
checkpoint = torch.load(model_name)

'''
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
'''

swa_model.load_state_dict(checkpoint["model_state_dict"]) #, strict=False)
swa_model.eval()

print("Loaded.")

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


img_path = "/home/compu/ymh/drawing/dataset/num_003/frame"
result_path = "/home/compu/ymh/drawing/dataset/num_004/image_semi"

images = os.listdir(img_path)
images.sort()
#print(images)

colors = np.array([[  0,   0,   0],
                   [  2,   6, 245],
                   [ 77, 165, 249],
                   [116, 252, 137],
                   [210,  51, 247],
                   [234,  52,  40],
                   [238, 121,  53],
                   [255, 250,  81]], dtype='uint8')
    
mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

for name in images:
    img = imread(os.path.join(img_path, name))
    img = transform(img)
    img = img.unsqueeze(0).float().to(device)
    
    output = swa_model(img)
    pred = output.argmax(dim=1).squeeze().detach().clone().cpu().numpy().astype(np.uint8)
    
    #np.save(os.path.join(result_path, "mask_%07d" % int(name.split("_")[1].split(".")[0])), pred)
    
    pic = torch.zeros(3, 256, 256, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = pred == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
    
    pic = pic.numpy().transpose(1,2,0)
    #plt.imshow()
    #plt.show()
    #plt.close()
    
    imsave(os.path.join(result_path, "pic_%07d.png" % int(name.split("_")[1].split(".")[0])), pic)
    
    print(name)
    
    
    