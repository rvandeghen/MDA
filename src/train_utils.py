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
import time
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from evaluate import Evaluator
from combine_utils import calibrate_posteriors


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def trainer(train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            metrics=False,
            max_epochs=1000,
            device='cpu'):

    logging.info("start training")

    best_loss = 9e99
    best_metric = -1
    
    os.makedirs(os.path.join(model_name), exist_ok=True)

    for epoch in range(max_epochs):
        best_model_path = os.path.join(model_name, "model.pth.tar")

        # train for one epoch
        loss_training = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            metrics=False,
            train=True,
            device=device)


        loss_validation = evaluate_model(
            val_loader,
            model, 
            criterion,
            epoch=epoch+1,
            metrics=metrics,
            device=device)

        logging.info("Validation loss at epoch " + str(epoch+1) + " -> " + str(loss_validation))

        is_best_loss = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        # Save the best model based on metric only if the evaluation frequency is short enough
        if is_best_loss:
            torch.save(state, best_model_path)

        # Learning rate scheduler update
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")

        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break

    return

def train_one_epoch(dataloader,
          model,
          criterion,
          optimizer=None,
          epoch=None,
          train=False,
          metrics=False,
          device='cpu'):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()
        
        if metrics:
            evaluator = Evaluator(model.num_classes)
    
    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, miniters=len(dataloader)//100) as t:
        for i, (images, labels, image_weight) in t:
            # measure data loading time
            data_time.update(time.time() - end)
    
    
            images = images.to(device)
            labels = labels.to(device)
            # compute output
            outputs = model(images)
            
            
            loss= criterion(outputs, labels)
    
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            
    
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            
            if metrics:
                if dataloader.dataset.calibrated:
                    outputs = calibrate_posteriors(outputs,dataloader.dataset.priors.cuda())
                evaluator.update(labels.cpu().numpy(),outputs.cpu().detach().numpy(), image_weight[0].item())
    
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)
    if not train and metrics:
        evaluator.accumulate()
        accuracy = evaluator.compute_accuracy()
        miou = evaluator.compute_miou()
        accuracy_balanced = evaluator.compute_accuracy(balanced = True)
        miou_balanced = evaluator.compute_miou(balanced = True)
        logging.info("Performance at epoch: " + str(epoch))
        logging.info("accuracy: " +  str(accuracy))
        logging.info("miou: " +  str( miou))
        logging.info("accuracy_balanced: " +  str(accuracy_balanced))
        logging.info("miou_balanced: " +  str( miou_balanced))

        return losses.avg, accuracy, miou, accuracy_balanced, miou_balanced, evaluator


    return losses.avg

def evaluate_model(dataloader,
          model,
          criterion,
          epoch,
          metrics,
          device='cpu',
          use_map=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to val mode

    model.eval()
    
    if metrics:
        evaluator = Evaluator(model.num_classes, map=use_map, priors=dataloader.dataset.priors)
    
    end = time.time()
    with torch.inference_mode():
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, miniters=len(dataloader)//100) as t:
            for i, (images, labels, image_weight) in t:
                # measure data loading time
                data_time.update(time.time() - end)

                desc = f'Evaluate {epoch}: '

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
    
                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                outputs = torch.nn.functional.softmax(model(images), dim=1)
        
                if dataloader.dataset.calibrated:
                    outputs = calibrate_posteriors(outputs,dataloader.dataset.priors.cuda(), model.num_classes)

                if metrics:
                    evaluator.update(labels.cpu().numpy(),outputs.cpu().detach().numpy(), image_weight[0].item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)

    if metrics:
        evaluator.accumulate()
        accuracy = evaluator.compute_accuracy()
        miou = evaluator.compute_miou()
        accuracy_balanced = evaluator.compute_accuracy(balanced = True)
        miou_balanced = evaluator.compute_miou(balanced = True)
        logging.info("Performance at epoch: " + str(epoch))
        logging.info("accuracy: " +  str(accuracy))
        logging.info("miou: " +  str( miou))
        logging.info("accuracy_balanced: " +  str(accuracy_balanced))
        logging.info("miou_balanced: " +  str( miou_balanced))

        return losses.avg, accuracy, miou, accuracy_balanced, miou_balanced, evaluator
    else:
        return losses.avg
