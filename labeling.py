import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import matplotlib.pyplot as plt

input_path = "C://유민형//개인 연구//animating//smp//joint//"
output_path = "C://유민형//개인 연구//animating//smp//mask//"

#input_path = "/home/compu/ymh/drawing/dataset/num_004/image_pseudo/"
#output_path = "/home/compu/ymh/drawing/dataset/num_004/onehot_pseudo/"

num_class = 18
width, height = 320, 180
ifresize = False
plt_show = True


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

files = os.listdir(input_path)
files.sort()

'''
import os
import torch
from skimage.io import imread, imsave
input_path = "C://유민형//개인 연구//animating//smp//color//"
files = os.listdir(input_path)
files.sort()
name = files[0]
index = int(name.split(".")[0])
target = imread(os.path.join(input_path, name))
target = torch.from_numpy(target[:,:,:3])
torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
'''

for name in files:
    index = int(name.split(".")[0])
    
    #index = int(name.split("_")[1].split(".")[0])
    
    target = imread(os.path.join(input_path, name))
    if plt_show:
        plt.imshow(target)
        plt.show()
        plt.close()
    
    if ifresize:
        target = resize(target,
                    (256,256),
                    mode='edge',
                    anti_aliasing=False,
                    anti_aliasing_sigma=None,
                    preserve_range=True,
                    order=0)
        target = target.astype('uint8')
    
        if plt_show:
            plt.imshow(target)
            plt.show()
            plt.close()
    
    target = torch.from_numpy(target[:,:,:3])
    a = target.reshape(-1,3)
    
    target = target.permute(2,0,1).contiguous()

    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

    mask = torch.zeros(height, width, dtype=torch.long)
    for k in mapping:
        #print(k)
        idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    
    #print(mask)
    mask = mask.unsqueeze(0).long()
    
    np.save(os.path.join(output_path, "mask_%07d" % index), mask.numpy())
    
    onehot = torch.zeros(num_class,height,width)
    onehot = onehot.scatter_(0,mask,1)
    onehot = onehot.numpy()
    
    #np.save(os.path.join(output_path, "onehot_%07d" % index), onehot)   
    
    onehot = torch.from_numpy(onehot)
    pred = onehot.argmax(dim=0).numpy().astype(np.uint8)
    
    pic = torch.zeros(3, height, width, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = pred == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
        
    pic = pic.numpy().transpose(1,2,0)
    if plt_show:
        plt.imshow(pic)
        plt.show()
        plt.close()
    
    b = pic[:,:,:3].reshape(-1,3)
    if np.unique(a,axis=0).shape[0]-np.unique(b,axis=0).shape[0] != 0:
        print(index)
        print(np.unique(a,axis=0).shape[0]-np.unique(b,axis=0).shape[0])

print("Done.")