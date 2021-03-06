import pickle

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import glob
from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    def __init__(self, data_root, image_size, paired=True, train=True, folder_A = "/real_A/", 
                 folder_B = "/fake_B/"):
        '''
        Parameters:

        '''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.data_path = data_root
        self.image_size = image_size
        # self.batch_size = batch_size
        self.folder_A = folder_A
        self.folder_B = folder_B
        print("path : ", self.data_path)
        self.train_names = glob.glob(self.data_path+self.folder_A+"*")
        print("train_names = ", len(self.train_names))
        self.names = [self.train_names[i].split('/')[-1] for i in range(len(self.train_names))]

        print("names : ", len(self.names))
        self.data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        # self.paired = paired

    def __len__(self):
        return len(self.names)

    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = image.convert("RGB")
        image = self.data_transforms(image).float()
        image = torch.autograd.Variable(image, requires_grad=False)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image[0].to(self.device)  #assumes that you're using GPU
    
    def __getitem__(self, idx):
        root = self.data_path

        images_dir = root + self.folder_A
        labels_dir = root + self.folder_B

        x = self.image_loader(images_dir + self.names[idx])
        y = self.image_loader(labels_dir + self.names[idx])

        return x,y


    

    #shift this to utils.py
    def show(self, img):
        npimg = img.cpu().detach().numpy()
        npimg = np.transpose(npimg, (1,2,0))
        if npimg.shape[2] == 3:
            plt.imshow(npimg)
        else:
            plt.imshow(npimg[:,:,0], cmap='gray')

    #shift this to utils.py
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().detach().numpy()
        plt.figure(figsize=(10,2))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto')

    def data_generator(self, iteration, train = True):
        root = self.data_path
        batch_size = self.batch_size

        images_dir = root + self.folder_A
        labels_dir = root + self.folder_B

        while True:
            x, y = [], []
          
            if train:
                start = iteration * batch_size
                end = min((iteration + 1) * batch_size, len(self.names))
                for i in range(start, end):
                    x.append(self.image_loader(images_dir + self.names[i]))
                    y.append(self.image_loader(labels_dir + self.names[i]))
            else:
                idx = np.random.choice(self.names, batch_size)
                for i in range(idx.shape[0]):
                    x.append(self.image_loader(images_dir + idx[i]))
                    y.append(self.image_loader(labels_dir + idx[i]))
       
            yield torch.stack(x), torch.stack(y)


class TestDataSet(Dataset):
    def __init__(self, data_root, image_size, paired=True, train=True, folder = "/test_images/"):
        '''
        Parameters:

        '''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.data_path = data_root
        self.image_size = image_size
        # self.batch_size = batch_size
        self.folder = folder
        print("path : ", self.data_path)
        self.test_names = glob.glob(self.data_path+self.folder+"*")
        print("train_names = ", len(self.test_names))
        self.names = [self.test_names[i].split('/')[-1] for i in range(len(self.test_names))]

        print("names : ", len(self.names))
        self.data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        # self.paired = paired

    def __len__(self):
        return len(self.names)

    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = image.convert("RGB")
        image = self.data_transforms(image).float()
        image = torch.autograd.Variable(image, requires_grad=False)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image[0].to(self.device)  #assumes that you're using GPU
    
    def __getitem__(self, idx):
        root = self.data_path

        images_dir = root + self.folder
        # labels_dir = root + self.folder_B

        x = self.image_loader(images_dir + self.names[idx])
        # y = self.image_loader(labels_dir + self.names[idx])

        return x