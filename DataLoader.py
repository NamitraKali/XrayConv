import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

class ImgDataset():
    def __init__(self, path, bs=1, aug=None, test_aug=None):
        self.path = path
        self.bs = bs
        self.aug = aug
        self.test_aug = test_aug
        
        self.train_dataset = datasets.ImageFolder(f'{path}/train', transform=aug)
        self.train = data.DataLoader(self.train_dataset, batch_size=bs, num_workers=2)
        
        try:
            self.test_dataset = datasets.ImageFolder(f'{path}/test',
                                                     transform=test_aug)
            self.test = data.DataLoader(self.test_dataset,
                                        batch_size=bs, num_workers=9)
        except:
            pass
        
        try:
            self.val_dataset = datasets.ImageFolder(f'{path}/val', transform=test_aug)
            self.val = data.DataLoader(self.val_dataset, batch_size=bs, num_workers=9)
        except:
            pass
        
    def apply_sampler(self, sampler):
        self.train_dataset = datasets.ImageFolder(f'{self.path}/train',
                                                  transform=self.aug)
        self.train = data.DataLoader(self.train_dataset, batch_size=self.bs,
                                     sampler=sampler, num_workers=2)