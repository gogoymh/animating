import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class multiclass_mask_cuda(Dataset):
    def __init__(self, path1, path2, device):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        self.device = device
        
        self.files = os.listdir(path2)
        self.files.sort()
        #print(self.files)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((180,320))
                ])  
        
        self.pil = transforms.ToPILImage()
        
        self.augmentation = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0.2,0.2))
                #transforms.RandomResizedCrop((256,256)),
                #transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]
        
        mask = np.load(os.path.join(self.segmantic_path, name))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        img = self.spatial_transform(img[:,:,:3])
        mask = mask.astype('uint8')
        mask = torch.from_numpy(mask)
        mask = self.pil(mask)
        
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        img = self.augmentation(img)
        torch.manual_seed(seed)
        random.seed(seed)
        mask = self.augmentation(mask)
        
        img = self.normalize(img)
        mask = self.basic_transform(mask) * 255
        
        return img.to(self.device), mask.squeeze().long().to(self.device)
        
    def __len__(self):
        return self.len

class multiclass_mask(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        
        self.files = os.listdir(path2)
        self.files.sort()
        #print(self.files)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((180,320))
                ])  
        
        self.pil = transforms.ToPILImage()
        
        self.augmentation = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0.2,0.2))
                #transforms.RandomResizedCrop((256,256)),
                #transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]
        
        mask = np.load(os.path.join(self.segmantic_path, name))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        img = self.spatial_transform(img[:,:,:3])
        mask = mask.astype('uint8')
        mask = torch.from_numpy(mask)
        mask = self.pil(mask)
        
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        img = self.augmentation(img)
        torch.manual_seed(seed)
        random.seed(seed)
        mask = self.augmentation(mask)
        
        img = self.normalize(img)
        mask = self.basic_transform(mask) * 255
        
        return img, mask.squeeze().long()
        
    def __len__(self):
        return self.len

class multiclass_onehot(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        
        self.files = os.listdir(path2)
        self.files.sort()
        #print(self.files)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256))
                ])  
        
        self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((256,256)),
                transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        #self.len = len(self.files)
        self.len = 1000
        print(self.files[-5:])
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]
        
        onehot = np.load(os.path.join(self.segmantic_path, name))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        img = self.spatial_transform(img[:,:,:3])
        onehot = torch.from_numpy(onehot)
        
        img = self.normalize(img)
        
        return img, onehot
        
    def __len__(self):
        return self.len

class multiclass_onehot_cuda(Dataset):
    def __init__(self, path1, path2, device):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        
        self.device = device
        
        self.files = os.listdir(path2)
        self.files.sort()
        #print(self.files)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256))
                ])  
        
        self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((256,256)),
                transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        #self.len = len(self.files)
        self.len = 1000
        print(self.files[-5:])
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]
        
        onehot = np.load(os.path.join(self.segmantic_path, name))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        img = self.spatial_transform(img[:,:,:3])
        onehot = torch.from_numpy(onehot)
        
        img = self.normalize(img)
        
        return img.float().to(self.device), onehot.float().to(self.device)
        
    def __len__(self):
        return self.len

class self_set(Dataset):
    def __init__(self, path1, train=True):
        super().__init__()
        
        self.img_path = path1
        
        self.files = os.listdir(path1)
        self.files.sort()
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        if train:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((180,320)),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomResizedCrop((426,240)),
                transforms.RandomAffine(0, shear=[-10, 10, -10, 10], translate=(0.1,0.1))
                ])  
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((180,320))
                ])  
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]        
        img = imread(os.path.join(self.img_path, name))

        img1 = self.spatial_transform(img)
        img2 = self.spatial_transform(img)
                
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        return img1, img2
        
    def __len__(self):
        return self.len

class ordinary_set(Dataset):
    def __init__(self, path1, device):
        super().__init__()
        
        self.img_path = path1
        self.device = device
        
        self.files = os.listdir(path1)
        self.files.sort()
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((180,320))
            ])  
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
                
        name = self.files[index]        
        img = imread(os.path.join(self.img_path, name))

        img = self.spatial_transform(img)
        img = self.normalize(img)
        
        return img.float().to(self.device)
        
    def __len__(self):
        return self.len

    
if __name__ == "__main__":
    
    path1 = "C://유민형//기타//영상//mp4//main_frame//"
    path2 = "C://유민형//기타//영상//mp4//smp//"
    
    import matplotlib.pyplot as plt
    import torch
    
    a = multiclass_mask(path1, path2)
    
    b, c = a.__getitem__(3)
    
    
    '''
    a = semantic_set(path1, path2,False)
    
    index = np.random.choice(30, 1)[0]
    #index = 0
    print(index)
    
    b, c = a.__getitem__(index)
    #print(b)
    #print(c)
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    c = c.squeeze().numpy()
    
    plt.imshow(c, cmap="gray")
    plt.show()
    plt.close()
    '''
    '''
    a = self_set(path1, True)
    
    index = np.random.choice(250, 1)[0]
    #index = 0
    print(index)
    
    b, c = a.__getitem__(index)
    #print(b)
    #print(c)
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
       
    c = c.numpy().transpose(1,2,0)
    
    plt.imshow(c)
    plt.show()
    plt.close()
    '''
    #print(b)
    #print(c)
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    colors = np.array([[  0,   0,   0],
                   [  6,   9,  49],
                   [ 61,  40,   6],
                   [ 72,  41,  43],
                   [ 92,  92, 135],
                   [153, 129, 115],
                   [199, 114, 121],
                   [251, 228, 215],
                   [255, 255, 255]], dtype='uint8')
    
    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
    
    pic = torch.zeros(3, 256, 256, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = c == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
        
    pic = pic.numpy().transpose(1,2,0)
    plt.imshow(pic)
    plt.show()
    plt.close()
    
    
    '''
    colors = np.array([[  0,   0,   0],
                   [  0,   0, 128],
                   [  0, 128,   0],
                   [  0, 128, 128],
                   [ 64,   0,   0],
                   [ 64, 128,   0],
                   [128,   0,   0],
                   [128,   0, 128],
                   [128, 128,   0],
                   [128, 128, 128],
                   [192,   0,   0],
                   [192, 128,   0]], dtype=np.uint8)
    
    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
    
    pred = torch.empty(3, 426, 240, dtype=torch.long)
    for i, k in enumerate(mapping):
        idx = c == i
        pred[0][idx] = k[0]
        pred[1][idx] = k[1]
        pred[2][idx] = k[2]
        
    plt.imshow(pred.numpy().transpose(1,2,0))
    plt.show()
    plt.close()
    '''
    
    
    
    