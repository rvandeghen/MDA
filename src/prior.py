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

import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from tqdm import tqdm

LABELS = {
	0: {'name':'road', 'color':[128,64,128]},
	1: {'name':'sidewalk', 'color':[244,35,232]},
	2: {'name':'building', 'color':[70,70,70]}, 
	3: {'name':'wall', 'color':[102,102,156]},
	4: {'name':'fence', 'color':[190,153,153]},
	5: {'name':'pole', 'color':[153,153,153]},
	6: {'name':'traffic light', 'color':[250,170,30]},
	7: {'name':'traffic sign', 'color':[220,220, 0]},
	8: {'name':'vegetation', 'color':[107,142,35]},
	9: {'name':'terrain', 'color':[152,251,152]},
	10: {'name':'sky', 'color':[70,130,180]},
	11: {'name':'person', 'color':[220,20,60]},
	12: {'name':'rider', 'color':[255, 0, 0]},
	13: {'name':'car', 'color':[0, 0,142]},
	14: {'name':'truck', 'color':[0,0,70]},
	15: {'name':'bus', 'color':[0,60,100]},
	16: {'name':'train', 'color':[0,80,100]},
	17: {'name':'motorcycle', 'color':[0, 0,230]},
	18: {'name':'bicycle', 'color':[119,11,32]},
}

def get_prior(dataset_path):
	# Check if prior.txt exists
	if os.path.exists(os.path.join(dataset_path,'prior.txt')):
		# If the file exists, return the tensor from the file txt
		with open(os.path.join(dataset_path,'prior.txt')) as file:
			txt_file = file.read()
		prior = txt_file.split('\n')
		prior = torch.tensor(list(map(float, prior[:-1])))
	else:
		# If the file does not exist, compute the prior for each class and return the tensor
		path = os.path.join(dataset_path, 'train', 'targets')

		prior = np.zeros(len(LABELS.keys()))
		targets_path = list()
		targets_path.extend(glob(path + '/*.npy'))

		for target_path in targets_path:
			target = np.load(target_path)
			target = target - 1

			for key, value in LABELS.items():
				count = (target == key).sum()
				prior[key] = int(prior[key] + count)

		prior = torch.tensor(prior)
		prior = torch.div(prior, prior.sum())
		np.savetxt(os.path.join(dataset_path,'prior.txt'), prior.numpy())
	return prior
