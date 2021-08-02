import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class joint_image(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.joint_path = path1
        self.img_path = path2
        
        self.files = os.listdir(path1)
        self.files.sort()
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
        self.pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            ])
        
        self.size_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256))
                ])
        
        self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((256,256)),
                transforms.RandomAffine(0, translate=(0.1,0.1), shear=[-10, 10, -10, 10])
                #transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    def __getitem__(self, index):
        name = self.files[index]
        
        joint = imread(os.path.join(self.joint_path, "pic_%07d.png" % int(name.split("_")[1].split(".")[0])))
        rand_joint = imread(os.path.join(self.joint_path, "pic_%07d.png" % np.random.choice(self.len, 1)[0]))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        joint = self.pil_transform(joint[:,:,:3])
        rand_joint = self.pil_transform(rand_joint[:,:,:3])
        img = self.size_transform(img[:,:,:3])
        
        seed = np.random.randint(2147483647)
        
        torch.manual_seed(seed)
        #random.seed(seed)
        joint = self.augmentation(joint)
        
        #torch.manual_seed(seed)
        #random.seed(seed)
        rand_joint = self.augmentation(rand_joint)
        
        torch.manual_seed(seed)
        #random.seed(seed)
        img = self.augmentation(img)
        
        joint = self.normalize(joint)
        rand_joint = self.normalize(rand_joint)
        img = self.normalize(img)
        
        return joint, rand_joint, img
        
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    path1 = "/data/ymh/drawing/dataset/num_004/image_semi/"
    path2 = "/data/ymh/drawing/dataset/num_003/frame/"
    save_path ="/data/ymh/drawing/save/num_005/"

    dataset_train = joint_image(path1, path2)
    
    a, b, c = dataset_train.__getitem__(0)
    
    a[0] = a[0]*0.5 + 0.5
    a[1] = a[1]*0.5 + 0.5
    a[2] = a[2]*0.5 + 0.5
       
    a = a.numpy().transpose(1,2,0)
    a = a * 255
    a = a.astype("uint8")
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    b = b * 255
    b = b.astype("uint8")
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
       
    c = c.numpy().transpose(1,2,0)
    c = c * 255
    c = c.astype("uint8")

    from skimage.io import imsave
    imsave("a.png", a)
    imsave("b.png", b)
    imsave("c.png", c)
    
    
    
    
