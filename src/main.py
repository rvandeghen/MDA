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
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from yaml.loader import BaseLoader
import torch
from train_model import train_model
from train_discriminator import train_discriminator
from combine import combine
from config_parser import Config

def main(args):

    f_config = open(args.config_file, 'r')
    config_data = Config(yaml.load(f_config, Loader=BaseLoader))

    # Fixing seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logging.info('number of domains: ' + str(len(config_data.datasets)))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
        
    if args.train_discriminator:
        train_discriminator(args, config_data, device)
        
    elif args.train_source_model:
        assert args.lambdas is not None and len(args.lambdas) == len(config_data.datasets), "The number of datasets: {} must be equal to the number of lambdas: {}".format(config_data.datasets, args.lambdas)
        assert sum(args.lambdas)==1, "Sum of lambdas must equal 1"
        train_model(args, config_data, device)
    
    elif args.combine:
        assert args.lambdas is not None and len(args.lambdas) == len(config_data.datasets), "The number of datasets: {} must be equal to the number of lambdas: {}".format(config_data.datasets, args.lambdas)
        assert sum(args.lambdas)==1, "Sum of lambdas must equal 1"
        combine(args, config_data, device)
        
    else:
        print('command not understood')
    

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Script to use MDA', formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Dataset
    parser.add_argument('--path', required=True, type=str,  help='path to the dataset as described in the readme')
    parser.add_argument('--config_file', required=True, type=str, help='config data file (e.g. driving.yaml or surveillance.yaml)')
    
    # Domain Mixture parameter
    parser.add_argument('--lambdas', default=None, type=float, nargs='+', help='List of lambdas to use (needed to train source models)')
    
    # Generic training parameters
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--batch_size', required=False, type=int,   default=4,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=10**(-3.5), help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )
    
    parser.add_argument('--save_name', type=str, default='model')
    
    # Source model training parameters
    parser.add_argument('--train_source_model', action='store_true', default=False, help='train a source model (or a combination of sources depending on lambdas)')
    
    # Discriminator training parameters
    parser.add_argument('--train_discriminator', action='store_true', default=False, help='train a discriminator')
    parser.add_argument('--mosaic_transform', action='store_true', default=False, help='use or not the mosaic transform during the discriminator training')
    
    # Combination parameters
    parser.add_argument('--source_models', default=None, type=str, nargs='+', help='List of source models')
    parser.add_argument('--discriminator', default=None, type=str, help='Discriminator model')
    parser.add_argument('--combine', action='store_true', default=False, help='assess the combination')
    parser.add_argument('--combination_name',   required=False, type=str,   default=None,     help='name of the combination' )
    parser.add_argument('--mle',  required=False, action='store_true', default=False,  help='Use the MLE as decision' )
    
    # General parameters
    parser.add_argument('--tiny',   required=False, type=int, default=None, help='only select a subset for debugging' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=8, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--seed', type=int, default = 0)

    args = parser.parse_args()

    # Logging information
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", args.save_name), exist_ok=True)
    log_path = os.path.join("models", args.save_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args)

    logging.info(f'Total Execution Time is {time.time()-start} seconds')