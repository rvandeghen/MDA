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
from tqdm import tqdm
from evaluate import Evaluator

def combine_confusion_matrix(dataloader, models, lambdas, device='cpu', use_mle=True):

    for model in models:
        model.eval()
    
    evaluator = Evaluator(models[0].num_classes)
    with torch.inference_mode():
        for m, l in zip(models, lambdas):

            tmp_evaluator = Evaluator(models[0].num_classes, mle=use_mle, priors=dataloader.dataset.priors)

            with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, miniters=len(dataloader)//100) as t:
                for i, (images, labels, image_weight) in t:
        
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = torch.nn.functional.softmax(m(images).squeeze(), dim=0).unsqueeze(0)
                
                    if dataloader.dataset.calibrated:
                        outputs = calibrate_posteriors(outputs,dataloader.dataset.priors.cuda())

                        tmp_evaluator.update(labels.cpu().numpy(),outputs.cpu().detach().numpy(), image_weight[0].item())

            evaluator.confusion_matrix += l * tmp_evaluator.confusion_matrix

    evaluator.accumulate()
    accuracy = evaluator.compute_accuracy()
    miou = evaluator.compute_miou()
    accuracy_balanced = evaluator.compute_accuracy(balanced = True)
    miou_balanced = evaluator.compute_miou(balanced = True)

    logging.info("Performance of naive combination: ")

    logging.info("accuracy: " +  str(accuracy))
    logging.info("miou: " +  str( miou))
    logging.info("accuracy_balanced: " +  str(accuracy_balanced))
    logging.info("miou_balanced: " +  str( miou_balanced))

    return accuracy, miou, accuracy_balanced, miou_balanced, evaluator

def combine_posteriors(dataloader, models, lambdas, device='cpu', use_mle=True):

    for model in models:
        model.eval()
    
    evaluator = Evaluator(models[0].num_classes, mle=use_mle, priors=dataloader.dataset.priors)
    with torch.inference_mode():
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, miniters=len(dataloader)//100) as t:
            for i, (images, labels, image_weight) in t:

                images = images.to(device)
                labels = labels.to(device)

                outputs = list()

                for model in models:

                    outputs.append(torch.nn.functional.softmax(model(images).squeeze(), dim=0))

                outputs = torch.stack(outputs)

                if dataloader.dataset.calibrated:
                    outputs = calibrate_posteriors(outputs,dataloader.dataset.priors.cuda())

                combine_output = torch.zeros_like(outputs[0])

                for (output, l) in zip(outputs, lambdas):
                    combine_output += l * output

                combine_output.unsqueeze_(0)
                        
                evaluator.update(labels.cpu().numpy(),combine_output.cpu().detach().numpy(), image_weight[0].item())

    evaluator.accumulate()
    accuracy = evaluator.compute_accuracy()
    miou = evaluator.compute_miou()
    accuracy_balanced = evaluator.compute_accuracy(balanced = True)
    miou_balanced = evaluator.compute_miou(balanced = True)

    logging.info("Performance of posteriors combination: ")

    logging.info("accuracy: " +  str(accuracy))
    logging.info("miou: " +  str( miou))
    logging.info("accuracy_balanced: " +  str(accuracy_balanced))
    logging.info("miou_balanced: " +  str( miou_balanced))

    return accuracy, miou, accuracy_balanced, miou_balanced, evaluator

def MDA(dataloader, discriminator, models, lambdas, device='cpu', use_mle=True):
    discriminator.eval()
    for model in models:
        model.eval()
        
    num_classes = models[0].num_classes
    evaluator = Evaluator(num_classes, mle=use_mle, priors=dataloader.dataset.priors)
    
    priors = dataloader.dataset.priors.to(device)
    priors = priors.unsqueeze(1).unsqueeze(1)
    lambdas_ = torch.tensor(lambdas)
    with torch.inference_mode():
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, miniters=len(dataloader)//100) as t:
            for i, (images, labels, image_weight) in t: 

                images = images.to(device)
                labels = labels.to(device)
                
                evidences = torch.nn.functional.softmax(discriminator(images).squeeze(), dim=0)
                lambdas = lambdas_.unsqueeze(-1).unsqueeze(-1).repeat(1, evidences.shape[1], evidences.shape[2]).to(device)
                evidences_calibrated = lambdas*evidences
              
                evidences_sum = evidences_calibrated.sum(dim=0)
               
                evidences_normalized = evidences_calibrated/evidences_sum.unsqueeze(0).repeat(evidences_calibrated.shape[0], 1, 1)     
                
                outputs = list()

                for model in models:

                    outputs.append(torch.nn.functional.softmax(model(images).squeeze(), dim=0))

                outputs = torch.stack(outputs)

                if dataloader.dataset.calibrated:
                    output_list = []
                    for (output, prior) in zip(outputs, dataloader.dataset.prior_list):
                        output = calibrate_posteriors(output.unsqueeze(0), prior.to(device))
                        output_list.append(output[0])
                    outputs = torch.stack(output_list)
                
                post = outputs*evidences_normalized.unsqueeze(1).repeat(1,num_classes,1,1)
          
                post_sum = post.sum(dim=0)
                post_target = post_sum
              
                post_target_normalized = post_target

                evaluator.update(labels.cpu().numpy(), post_target_normalized.unsqueeze(0).cpu().detach().numpy(), image_weight[0].item())

    evaluator.accumulate()
    accuracy = evaluator.compute_accuracy()
    miou = evaluator.compute_miou()
    accuracy_balanced = evaluator.compute_accuracy(balanced = True)
    miou_balanced = evaluator.compute_miou(balanced = True)

    logging.info("Performance of our combination: ")
    
    logging.info("accuracy: " +  str(accuracy))
    logging.info("miou: " +  str( miou))
    logging.info("accuracy_balanced: " +  str(accuracy_balanced))
    logging.info("miou_balanced: " +  str( miou_balanced))

    return accuracy, miou, accuracy_balanced, miou_balanced, evaluator
   
def calibrate_posteriors(posteriors, priors, u=1):

    new_posteriors = list()
    for posterior in posteriors:
        posterior = posterior * (priors.unsqueeze(-1).unsqueeze(-1)).repeat(1, posterior.shape[1],posterior.shape[2]) * u
        posterior = posterior / (posterior.sum(dim=0).unsqueeze(0).repeat(posterior.shape[0],1,1))
        new_posteriors.append(posterior)
    return torch.stack(new_posteriors)