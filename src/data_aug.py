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

import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Mosaic:
    """Mosaic augmentation.
    Greatly inspired from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/transforms.py
    
    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
                           mosaic transform
                           
                                center_x
              
                     +--------------+----------+
                     |              |          |
                     |  image1      |  image2  |
                     |              |          |
                     |              |          |
        center_y     +--------------+----------+
                     |              |          |
                     |  image3      |  image4  |
                     |              |          |
                     +--------------+----------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch
    Args:
        img_scale (Sequence[int]): Image size. The shape order should be (height, width).
            Default to (720, 1280).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output (randomly sampled). Default to (0.01, 0.99).
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(self,
                 img_scale=(720, 1280),
                 center_ratio_range=(0.01,0.99),
                 prob=1.0):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'

        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.prob = prob

    def __call__(self, images, domains):

        if random.uniform(0, 1) > self.prob:
            return images[0], domains[0]
        image, domain = self._mosaic_transform(images, domains)
        return image, domain

    def _mosaic_transform(self, images, domains):

        
        new_image = torch.zeros((3,)+(self.img_scale))
        new_domain = torch.zeros((self.img_scale),dtype=torch.long)
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_y, center_x)
        # image1, image2, image3, image4
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            image, domain = images[i], domains[i]
            if loc=='top_left':
                size = (center_y, center_x)
                pos = (0,0)
            elif loc == 'top_right':
                size = (center_y, self.img_scale[1] - center_x)
                pos = (0,center_x)
            elif loc == 'bottom_left':
                size = (self.img_scale[0] - center_y, center_x)
                pos = (center_y, 0)
            elif loc == 'bottom_right':
                size = (self.img_scale[0]-center_y, self.img_scale[1]-center_x)
                pos = (center_y, center_x)
            
            crop_params = T.RandomCrop.get_params(image, (size[0], size[1]))
            
            crop_image = F.crop(image, *crop_params)
            crop_domain = F.crop(domain, *crop_params)
 
            new_image[:,pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]] = crop_image
            new_domain[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1]] = crop_domain
        
        return new_image, new_domain
    