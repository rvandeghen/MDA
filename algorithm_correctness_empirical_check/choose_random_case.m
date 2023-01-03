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
% This function chooses one random case on which the algorithm can be
% tested. The number of evidences is between 2 and 10, the  number of
% hypotheses between 2 and 10, the number of source domains also between
% 2 and 10. Moreover, the probability measures in the source domains are
% randomly chosen, as well as the mixture weights defining the target
% domain.
% 
% This function returns:
% - distribution_EH_given_S, the chosen join distribution of evidences and
%   hypotheses in each source domain;
% - true_sources_priors, the mixture weights defining the target domain;
% - true_target_posteriors, the groundtruth posteriors in the target
%   domain, for each evidence. The goal of the unsupervised domain
%   adaptation algorithm is to retrieve this, up to a numerical tolerance.

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ distribution_EH_given_S , true_sources_priors , true_target_posteriors ] = choose_random_case ()

    % Generate a random problem (no matter if the generator is uniform or
    % not). It is given by the join distribution of evidences, hypothesis,
    % and source domains. In the paper, this distribution is modelled as
    % the probability measure P introduced in Section 3.3.2. The first
    % dimension is for the evidences, the second for the hypotheses, and
    % the third for the source domains.

    min_num_E = 2 ;
    max_num_E = 10 ;
    num_E = min_num_E + floor ( rand () * ( max_num_E - min_num_E + 1 ) ) ;
    
    min_num_H = 2 ;
    max_num_H = 10 ;
    num_H = min_num_H + floor ( rand () * ( max_num_H - min_num_H + 1 ) ) ;
    
    min_num_S = 2 ;
    max_num_S = 10 ;
    num_S = min_num_S + floor ( rand () * ( max_num_S - min_num_S + 1 ) ) ;

    distribution_EHS = rand ( [ num_E , num_H , num_S ] ) ;
    distribution_EHS = distribution_EHS / sum ( distribution_EHS ( : ) ) ; % P ( E , H , S )
    
    % Compute the source priors, e.g. the mixture weights

    true_sources_priors = marginalize_H ( marginalize_E ( distribution_EHS ) ) ; % P ( S ) = lambdas
    
    % Compute the expected target posteriors.
    
    true_target_distribution = marginalize_S ( distribution_EHS ) ; % P ( E , H )
    true_target_posteriors = true_target_distribution ./ marginalize_H ( true_target_distribution ) ; % P ( H | E )

    % Forget the source priors.

    distribution_EH_given_S = distribution_EHS ./ true_sources_priors ;

end
