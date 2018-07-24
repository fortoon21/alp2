from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class V_Caption_Patch_Alp2(data.Dataset):
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

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            label_origin = int(splited[1])
            label = []
            if label_origin <= 25:
                label.append(label_origin)
                label.append(0)
            else:
                label.append(label_origin - 26)
                label.append(1)

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

        label_alp = self.labels[idx][0].clone()
        label_sb = self.labels[idx][1].clone()

        if self.transform:
            img = self.transform(img)

        return img, (label_alp, label_sb)

    def __len__(self):
        return self.num_imgs