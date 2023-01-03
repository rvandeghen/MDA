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
from dataset import ModelDataset

def train_model(args, config_data, device):
    
    dataset_train = ModelDataset(args.path, config_data, args.lambdas, split="train", transforms=None)
    dataset_valid = ModelDataset(args.path, config_data, args.lambdas, split="val", test_mode=True, transforms=None)
    
    # Create the segmentation model
    model = TinyNet(num_classes=config_data.num_classes).to(device)
    
    # Logging information about the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))
    
    # Create the dataloaders for train, validation and test 
    train_loader = torch.utils.data.DataLoader(dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.max_num_worker, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_valid,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.max_num_worker, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss(weight=dataset_train.weights.to(device), ignore_index=-1)
 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                betas=(0.9, 0.999), eps=1e-06, 
                                weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                           verbose=True, patience=args.patience)
    
    # Start training
    trainer(train_loader, val_loader, 
            model, optimizer, scheduler, criterion,
            model_name=args.save_name,
            max_epochs=args.max_epochs,
            metrics=False,
            device=device)
