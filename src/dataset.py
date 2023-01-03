"""
----------------------------------------------------------------------------------------
Copyright (c) 2023 - see AUTHORS file
This file is part of the MDA software.
This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this 
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import numpy as np
import random
import os
import torch
from natsort import natsorted
from glob import glob
import numpy as np
import random
import sys


class DiscriminatorDataset(Dataset):
    def __init__(self, path, config, split="train", test_mode=False, transforms_instance=None, transforms_combination=None):
        self.path = path
        self.datasets = config.datasets
        self.n_domains = len(config.datasets)
        self.substract_background = config.substract_background
        self.img_size = config.img_size
        self.split = split
        self.test_mode = test_mode
        self.transforms_instance = transforms_instance
        self.transforms_combination = transforms_combination
        self.calibrated = False # not needed for training, calibration is done during the combination phase

        self.image_list = []
        self.domain_list = []

        for i, name in enumerate(self.datasets):
            dataset = natsorted(glob(os.path.join(self.path, name, self.split, "images", "*")))
            self.image_list.extend(dataset)
            self.domain_list.extend([i]*len(dataset))
  
    def __getitem__(self, index):
        if self.substract_background:
            sub = 1
        else:
            sub = 0
        
        image_background = torchvision.io.read_image(self.image_list[index])/255.
        if self.transforms_instance:
            image_background = self.transforms_instance(image_background)
        domain_background = (torch.ones(self.img_size)*self.domain_list[index]).long()
    
        if self.test_mode:
            return image_background, domain_background, torch.Tensor([1.])
        class_background = torch.from_numpy(np.load(self.image_list[index].replace("images","targets")[:-4] + ".npy")).type(torch.LongTensor)-sub

        if self.transforms_combination:
            images = [image_background]
            domains = [domain_background]
            for n in range(4):
                idx_foreground = random.randint(0,self.__len__()-1)
                img = torchvision.io.read_image(self.image_list[idx_foreground])/255.
                if self.transforms_instance:
                    img = self.transforms_instance(img)
                images.append(img)
                domains.append((torch.ones(self.img_size)*self.domain_list[idx_foreground]).long())
            combined_image, combined_domain = self.transforms_combination(images, domains) 
            return combined_image, combined_domain, torch.Tensor([1.])
        else:
            return image_background, domain_background, torch.Tensor([1.])
        
    def __len__(self,):
        return len(self.image_list)


    
class ModelDataset(Dataset):
    def __init__(self,path, config, lambdas, split="train", test_mode=False, transforms=None):
        self.path = path
        self.datasets = config.datasets
        self.split = split
        self.lambdas = lambdas
        self.substract_background = config.substract_background
        self.num_classes = config.num_classes
        self.test_mode = test_mode
        self.priors, self.prior_list = self.get_priors()
        self.weights = None if test_mode else 1/self.priors
        self.calibrated = True
        self.transforms = transforms

        if test_mode:
            self.image_list = []
            self.image_weights = []
            for (name, l) in zip(self.datasets, self.lambdas):
                self.image_list.extend(natsorted(glob(os.path.join(self.path, name, self.split, "images", "*"))))
                self.image_weights.extend([l]*config.max_size[split])
        else:
            random.seed(0)
            self.image_list = []
            for (name, l) in zip(self.datasets, self.lambdas):
                tmp_list = natsorted(glob(os.path.join(self.path, name, self.split, "images", "*")))
                random.shuffle(tmp_list)
                self.image_list.extend(tmp_list[:int(config.max_size[split]*l)])
            

    def get_priors(self,):
        priors = 0
        prior_list = []
        for (name, l) in zip(self.datasets, self.lambdas):
            priors += get_prior(os.path.join(self.path, name))*l
            prior_list.append(get_prior(os.path.join(self.path, name)))
        return priors, prior_list

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.image_list[index])/255.
        label = torch.from_numpy(np.load(self.image_list[index].replace("images","targets")[:-4] + ".npy")).type(torch.LongTensor)
        if self.substract_background:
            label -= 1
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        if self.test_mode:
            image_weight = self.image_weights[index]
            return image, label, image_weight
        return image, label, torch.Tensor([1.])

    def __len__(self,):
        return len(self.image_list)

def get_prior(dataset_path):
    # Check if prior.txt exists
    if os.path.exists(os.path.join(dataset_path,'prior.txt')):
        # If the file exists, return the tensor from the file txt
        with open(os.path.join(dataset_path,'prior.txt')) as file:
            txt_file = file.read()
        prior = txt_file.split('\n')
        prior = torch.tensor(list(map(float, prior[:-1])))
        return prior
    else:
        print('Priors do not exist')
        sys.exit()
