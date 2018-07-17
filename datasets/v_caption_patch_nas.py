from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class V_Caption_Patch_NAS(data.Dataset):
    '''Load image/labels/boxes from a list file.
    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        offset=0
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            if line.startswith('bgnum') or line.startswith('num'):
                offset=0
            elif line.startswith('bgalp') or line.startswith('alp'):
                offset=10
            elif line.startswith('bgsym') or line.startswith('sym'):
                offset=62

            label = []
            label.append(int(splited[1]) + offset)
            self.labels.append(torch.LongTensor(label))


    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        labels = self.labels[idx][0].clone()

        if self.transform:
            img = self.transform(img)

        return img, labels

    def __len__(self):
        return self.num_imgs