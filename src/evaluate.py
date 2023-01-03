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

import numpy as np
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, n_classes=19, mle=False, priors=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.mle = mle
        self.priors = priors
        
    def update(self, targets, predictions, weight=1.):
        if self.mle:
            predictions = (predictions.squeeze(0)/self.priors.unsqueeze(-1).unsqueeze(-1).repeat(1, predictions.shape[2], predictions.shape[3])).unsqueeze(0).numpy()
            predictions = np.argmax(predictions, axis=1)            
        else:
            predictions = np.argmax(predictions, axis=1)
        idx = targets != -1
        targets, predictions = targets[idx], predictions[idx]
        self.confusion_matrix += weight*confusion_matrix(targets, predictions, labels=np.arange(self.n_classes))

    def accumulate(self):
        self.idxs = np.where(self.confusion_matrix.sum(axis=0) != 0)[0]
        self.confusion_matrix = self.confusion_matrix[self.idxs, :][:, self.idxs]
        
    def compute_balanced_matrix(self):
        return self.confusion_matrix/np.expand_dims(self.confusion_matrix.sum(axis=1), 1).repeat(len(self.idxs), 1)
        
    def compute_accuracy(self, balanced=False):
        
        mat = self.compute_balanced_matrix() if balanced else self.confusion_matrix
        
        return mat.diagonal().sum()/mat.sum()
    
    def compute_iou_c(self, cls, balanced=False):
        
        mat = self.compute_balanced_matrix() if balanced else self.confusion_matrix
        
        return mat[cls, cls] / (mat[cls, :].sum() + mat[:, cls].sum() - mat[cls, cls])
    
    def compute_miou(self, balanced=False):
        
        miou = 0.
        
        for i in range(len(self.idxs)):
            miou += self.compute_iou_c(i, balanced)
            
        return miou/self.n_classes