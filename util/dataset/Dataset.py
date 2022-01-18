import os.path

import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import cv2
from PIL import Image

class ImageSet(VisionDataset):
    def __init__(self, data, opt,phase,root):
        self.data = data
        self.opt=opt
        super(ImageSet, self).__init__('')
        self.root=root
        if phase=='train':
            self.transform = transforms.Compose([
                transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                         transforms.Resize(opt.loadSize, interpolation=2),
                                         transforms.Resize(opt.loadSize, interpolation=3),
                                         transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                         transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                         transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                         transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                         transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path=os.path.join(self.root,self.data[idx])
        img = cv2.imread(path)
        h,w=img.shape[0],img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img,h,w
