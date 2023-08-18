"""This is dataset.py from pytorch-examples.

Refer to

https://github.com/pytorch/examples/blob/master/super_resolution/dataset.py.
"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import matplotlib.pyplot as plt

from os import listdir
from os.path import join

from PIL import Image


def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _load_img(filepath, RGB=True):
    img = Image.open(filepath)
    #print(img)
    if RGB:
        pass
    else:
        img = img.convert('YCbCr')
        img, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    """Generate an image-to-image dataset from images from the given folder."""

    def __init__(self, image_dir, replicate=1, input_transform=None, target_transform=None, RGB=True, noise_level=0.0):
        """Init with directory, transforms and RGB switch."""
        super(DatasetFromFolder, self).__init__()
        all_img = {}
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)]
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        with open("/public/home/neic/LC/NUS-WIDE/Groundtruth/nus_wide_train_imglist-1.txt", "r") as f:
            all_img = f.readlines()
        for i in range(len(all_img)):
            all_img[i] = all_img[i].replace('\n', '') 
        self.image_filenames = all_img
        #print(f'filename = {self.image_filenames}')
        
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.replicate = replicate
        self.classes = [None]
        self.RGB = RGB
        self.noise_level = noise_level

    def __getitem__(self, index):
        """Index into dataset."""
        print(f'id = {index % len(self.image_filenames)}')
        print(f'len = {len(self.image_filenames)}')
        #print(f'name = {self.image_filenames}')
        print(f'select_file_name = {self.image_filenames[index % len(self.image_filenames)]}')
        print(f'num={index // len(self.image_filenames)}')

       
        img_names = self.image_filenames[index % len(self.image_filenames)]
        print(f'img_names = {img_names}')
        img_site = index % len(self.image_filenames)
        
        #print(f'img_names={img_names}')
        #print(f'sure = {index % len(img_names)}')
        #print(f'img =  {img_names[index % len(img_names)]}')

        #print(f'input= {osp.join(self.image_filenames[index // len(self.image_filenames)], img_names[index % len(img_names)])}')
        #print(f'name= {self.image_filenames[index % len(self.image_filenames)]}')
        #input = _load_img(self.image_filenames[index % len(self.image_filenames)], RGB=self.RGB)
        #input = _load_img(osp.join(self.image_filenames[index // len(self.image_filenames)], img_names[index]), RGB=self.RGB)
        input = _load_img((img_names), RGB=self.RGB)
        Full_file = img_names
        
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
            #nput = input.resize(256,256)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise_level > 0:
            # Add noise
            input += self.noise_level * torch.randn_like(input)

        each_len = {}
        for i in range (len(self.image_filenames)):
            each_len[i] = (len(self.image_filenames[i]))
        #print(f'each_len = {each_len}')
        #返回  x, x, 
        #img_names = 文件夹中所有图片名，
        #img_site = 图片在文件夹中序列,
        #index // len(self.image_filenames) =图片所在文件夹序列，
        #self.image_filenames[index // len(self.image_filenames)] = 图片所在文件夹名称
        #each_len=每一个文件夹内图片数量
        #Full_file=图片全部文件路径
        #print(f'img = {img_names[index % len(img_names)]}')
        #print(f'index % len(self.image_filenames) = {index % len(self.image_filenames)}')
        #print(Full_file)
        return input, target, img_names, img_site, index % len(self.image_filenames), self.image_filenames[index % len(self.image_filenames)], each_len, Full_file
        #return input, target, img_names, img_site, img_names[index % len(img_names)], self.image_filenames[index % len(self.image_filenames)], each_len, Full_file

    def __len__(self):
        """Length is amount of files found."""
        return len(self.image_filenames) * self.replicate
