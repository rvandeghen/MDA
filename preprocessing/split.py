import os
import shutil
from natsort import natsorted
from glob import glob
import numpy as np
import random
import itertools
from tqdm import tqdm

Root = './'
Datasets = ['CityScapes', 'BDD100K']
# Datasets = ['Drone', 'M-30-HD', 'RainSnow', 'Sherbrooke']

# Define the number of samples for train/val/test set
nb_images_train = 2759 # 1299
nb_images_val = 250 # 300
nb_images_test = 500 # 300

for dataset in Datasets :
	# Get the path of all the images of the dataset
	files_path = []
	files_path.extend(natsorted(glob(os.path.join(Root, dataset, '*.jpg'))))
	files_path.extend(natsorted(glob(os.path.join(Root, dataset, '*.png'))))

	# Define a list that has the same length as the dataset; each element is equal to 1 for training set, 2 for validation set, and 3 for test set, and 4 for nothing
	split = np.ones_like(files_path, dtype=int)
	split[nb_images_train:nb_images_train+nb_images_val] = 2
	split[nb_images_train+nb_images_val:nb_images_train+nb_images_val+nb_images_test] = 3
	split[nb_images_train+nb_images_val+nb_images_test:] = 4
	
	# Shuffle randomly the list to obtain random sets of train, val, test
	random.seed(0)
	random.shuffle(split)

	for (value, path) in tqdm(zip(split, files_path)):
		filename = os.path.basename(path)
		if value == 1: # train
			shutil.copy(path, os.path.join(Root, dataset, 'train/images/', filename))
		elif value == 2: # val
			shutil.copy(path, os.path.join(Root, dataset, 'val/images/', filename))
		elif value == 3: # test
			shutil.copy(path, os.path.join(Root, dataset, 'test/images/', filename))

