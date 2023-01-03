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

import torch
import copy
import logging
from dataset import ModelDataset
from network import TinyNet
from train_utils import evaluate_model
from combine_utils import combine_confusion_matrix, combine_posteriors, MDA

def combine(args, config_data, device):

    dataset_test = ModelDataset(args.path, config_data, args.lambdas, split="test", test_mode=True, transforms=None)
    test_loader = torch.utils.data.DataLoader(dataset_test,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.max_num_worker, pin_memory=True)

    # Load the best models and compute their performance

    model = TinyNet(num_classes=config_data.num_classes).to(device)

    model_list = []
    for m in args.source_models:
        checkpoint = torch.load(m, map_location=device)
        tmp_model = copy.deepcopy(model)
        tmp_model.load_state_dict(checkpoint['state_dict'])
        model_list.append(tmp_model)
        
        
    if args.combination_name is None:

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss, accuracy, miou, accuracy_balanced, miou_balanced, evaluator = evaluate_model(test_loader, model=model_list[0], criterion=criterion, epoch='End', metric=True, device=device, use_mle=args.mle)

        logging.info("Best performance at end of training ")
        logging.info("loss: " +  str(loss))
        logging.info("accuracy: " +  str(round(100*accuracy, 1)))
        logging.info("miou: " +  str(round(100*miou, 1)))
        logging.info("accuracy_balanced: " +  str(round(100*accuracy_balanced, 1)))
        logging.info("miou_balanced: " +  str(round(100*miou_balanced, 1)))

        return loss, accuracy, miou, accuracy_balanced, miou_balanced, evaluator

    if args.combination_name == 'naive':
        accuracy, miou, accuracy_balanced, miou_balanced, evaluator = combine_confusion_matrix(test_loader, model_list, args.lambdas, device=device, use_mle=args.mle)

        logging.info("Performance for combination of confusion matrix ")
        logging.info("accuracy: " +  str(round(100*accuracy, 1)))
        logging.info("miou: " +  str(round(100*miou, 1)))
        logging.info("accuracy_balanced: " +  str(round(100*accuracy_balanced, 1)))
        logging.info("miou_balanced: " +  str(round(100*miou_balanced, 1)))

        return None, accuracy, miou, accuracy_balanced, miou_balanced, evaluator

    if args.combination_name == 'posteriors':
        accuracy, miou, accuracy_balanced, miou_balanced, evaluator = combine_posteriors(test_loader, model_list, args.lambdas, device=device, use_mle=args.mle)

        logging.info("Performance for combination of posteriors ")
        logging.info("accuracy: " +  str(round(100*accuracy, 1)))
        logging.info("miou: " +  str(round(100*miou, 1)))
        logging.info("accuracy_balanced: " +  str(round(100*accuracy_balanced, 1)))
        logging.info("miou_balanced: " +  str(round(100*miou_balanced, 1)))

        return None, accuracy, miou, accuracy_balanced, miou_balanced, evaluator

    if args.combination_name == 'ours':
        discriminator = TinyNet(num_classes=len(args.lambdas)).to(device)
        checkpoint = torch.load(args.discriminator, map_location=device)
        discriminator.load_state_dict(checkpoint['state_dict'])

        accuracy, miou, accuracy_balanced, miou_balanced, evaluator = MDA(test_loader, discriminator, model_list, args.lambdas, device=device, use_mle=args.mle)

        logging.info("Performance for MDA combination ")
        logging.info("accuracy: " +  str(round(100*accuracy, 1)))
        logging.info("miou: " +  str(round(100*miou, 1)))
        logging.info("accuracy_balanced: " +  str(round(100*accuracy_balanced, 1)))
        logging.info("miou_balanced: " +  str(round(100*miou_balanced, 1)))

        return None, accuracy, miou, accuracy_balanced, miou_balanced, evaluator