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

import logging
import torch
from train_utils import trainer 
from network import TinyNet
from dataset import DiscriminatorDataset
from data_aug import Mosaic


def train_discriminator(args, config_data, device):
    
    if args.mosaic_transform:
        transform_combination = Mosaic(prob=0.5)
    else:
        transform_combination = None
    
    # Create Train Validation and Test datasets
    dataset_train_disc = DiscriminatorDataset(args.path, config_data, split="train", transforms_instance=None, transforms_combination=transform_combination)
    dataset_valid_disc = DiscriminatorDataset(args.path, config_data, split="val", test_mode=True, transforms_instance=None, transforms_combination=None)

    # Create the Pixel Discrimator
    disc = TinyNet(num_classes=len(config_data.datasets)).to(device)
    
    # Logging information about the discriminator
    total_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in disc.parameters() if p.requires_grad]
    logging.info("Total number of parameters of discriminator: " + str(total_params))
    
    
    # Create the dataloaders for train, validation and test 
    train_loader_disc = torch.utils.data.DataLoader(dataset_train_disc,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.max_num_worker, pin_memory=True)
    val_loader_disc = torch.utils.data.DataLoader(dataset_valid_disc,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.max_num_worker, pin_memory=True)
   
    criterion_disc = torch.nn.CrossEntropyLoss()
   
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=args.LR, 
                                betas=(0.9, 0.999), eps=1e-06, 
                                weight_decay=0, amsgrad=False)
    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, 'min', 
                                                                verbose=True, patience=args.patience)

    # Start discriminator training 
    trainer(train_loader_disc, val_loader_disc, 
            disc, optimizer_disc, scheduler_disc, criterion_disc,
            model_name= args.save_name,
            max_epochs=args.max_epochs,
            metrics=False,
            device=device)
