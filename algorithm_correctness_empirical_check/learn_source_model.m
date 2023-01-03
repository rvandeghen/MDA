%{
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
%}
% This function computes the exact source model for any given
% join distribution of evidences and hypotheses. This model gives the
% distribution of hypothesis for each possible evidence, that is the
% posteriors in the source domain.
% 
% If the join distribution of evidences and hypotheses given in
% argument is shifted, then the model is "exact up to a target shift".

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ model_hypotheses_posteriors ] = learn_source_model ( source_distribution )
    % PSk ( H | E , S = Sk ) = PSk ( E , H | S = Sk ) / PSk ( E | S = Sk )
    model_hypotheses_posteriors = source_distribution ./ marginalize_H ( source_distribution ) ;
end
