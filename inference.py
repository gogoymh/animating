import os
import torch
from torchvision import transforms
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

#from my_conv_unet import my_unet, Unet_last
from my_conv_unet import my_unet_full

n_class = 18
height, width = 180, 320

device = torch.device("cuda:0")

model = my_unet_full(3,64,n_class).to(device)

model_name = os.path.join("/data/ymh/drawing/save/num_006/Drawing/character", "swa_finetune.pth")
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

model.load_state_dict(new_state_dict)#, strict=False)
'''
model.load_state_dict(checkpoint["model_state_dict"])#, strict=False)
model.eval()
'''
model_part1 = my_unet(3,64,n_class).to(device)
model_part2 = Unet_last(n_class).to(device)


save_path = "/data/ymh/drawing/save/num_006/Drawing/character"

model_name_1 = os.path.join(save_path, "base_part1.pth")
model_name_2 = os.path.join(save_path, "base_part2.pth")

checkpoint1 = torch.load(model_name_1)
checkpoint2 = torch.load(model_name_2)
'''
'''
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
'''
'''
model_part1.load_state_dict(checkpoint1["model_state_dict"])
model_part2.load_state_dict(checkpoint2["model_state_dict"])

model_part1.eval()
model_part2.eval()
'''
print("Loaded.")

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((height,width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


img_path = "/data/ymh/drawing/dataset/num_006/frame"
result_path = "/data/ymh/drawing/dataset/num_006/image_pseudo"

images = os.listdir(img_path)
images.sort()
print(images[:5])


colors = np.array([[  0,   0,   0],
                   [  2,   6, 245],
                   [ 35,  89,  30],
                   [ 77, 165, 249],
                   [110,  70, 227],
                   [116, 252, 137],
                   [144,  76,  47],
                   [154, 224, 214],
                   [201, 242, 131],
                   [206,  46, 113],
                   [206, 123, 213],
                   [210,  51, 247],
                   [214, 136, 145],
                   [228, 121, 127],
                   [234,  52,  40],
                   [238, 121,  53],
                   [245, 191,  63],
                   [255, 250,  81]], dtype="uint8")
    
mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

for name in images[:1000]:
    img = imread(os.path.join(img_path, name))
    img = transform(img)
    img = img.unsqueeze(0).float().to(device)
    
    output = model(img)
    pred = output.argmax(dim=1).squeeze().detach().clone().cpu().numpy().astype(np.uint8)
    
    #np.save(os.path.join(result_path, "mask_%07d" % int(name.split("_")[1].split(".")[0])), pred)
    
    pic = torch.zeros(3, height, width, dtype=torch.uint8)
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
    
    
    