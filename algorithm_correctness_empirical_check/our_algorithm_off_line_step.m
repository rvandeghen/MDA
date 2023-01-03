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
% This function provides an easy to read implementation of the "off-line
% step" of the unsupervised domain adaptation algorithm presented in
% Section 3.4 of the paper, for the mixture domain adaptation problem
% specified in Section 3.2.
%
% During the off-line step, a source model (exact up to a target shift)
% is learned for each source domain, and a domain discriminator model
% (also exact up to a target shift) is also learned. Unless a second
% argument is given and set to "true", the shifts are chosen at random.
% 
% This function returns:
% - true_hypotheses_priors: the probabilities of the various hypotheses in
%   each source domain;
% - assumed_hypothesis_priors: the reference used by the source models to
%   express the posteriors they give in output;
% - source_models: the learned source models (exact up to a target shift);
% - assumed_sources_priors: the reference used by the domain discriminator
%   model to express the probabilities it gives in output;
% - domain_discriminator_model: the learned domain discriminator model
%   (exact up to a target shift).

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ true_hypotheses_priors , assumed_hypothesis_priors , source_models , assumed_sources_priors , domain_discriminator_model ] = our_algorithm_off_line_step ( distribution_EH_given_S , uniform )
    
    if nargin < 2
        uniform = false ;
    end
    if uniform
        disp ( 'WARNING: test ony valid in UNIFROM mode!' )
    end

    num_E = size ( distribution_EH_given_S , 1 ) ;
    num_H = size ( distribution_EH_given_S , 2 ) ;
    num_S = size ( distribution_EH_given_S , 3 ) ;
    
    % Determine the true priors of hypotheses and arbitrarilly choose
    % the assumed priors of hypotheses in all source domains.
    
    true_hypotheses_priors = nan ( [ 1 , num_H , num_S ] ) ;
    assumed_hypothesis_priors = nan ( [ 1 , num_H , num_S ] ) ;
    for k = 1 : num_S
        source_distribution = distribution_EH_given_S ( : , : , k ) ; % P ( E , H | S = Sk )
        true_priors = marginalize_E ( source_distribution ) ; % P ( H | S = Sk )
        true_hypotheses_priors ( : , : , k ) = true_priors ;
        if uniform
            assumed_priors = ones ( [ 1 , num_H , 1 ] ) / num_H ;
        else
            assumed_priors = rand ( [ 1 , num_H , 1 ] ) ;
            assumed_priors = assumed_priors ./ sum ( assumed_priors ) ;
        end
        assumed_hypothesis_priors ( : , : , k ) = assumed_priors ;
    end
    
    % Learn the source models
    
    source_models = nan ( [ num_E , num_H , num_S ] ) ;
    for k = 1 : num_S
        true_source_distribution = distribution_EH_given_S ( : , : , k ) ; % P ( E , H | S = Sk )
        true_priors = true_hypotheses_priors ( : , : , k ) ;
        assumed_priors = assumed_hypothesis_priors ( : , : , k ) ;
        assumed_source_distribution = true_source_distribution .* assumed_priors ./ true_priors ; % PSk ( E , H | S = Sk )
        model = learn_source_model ( assumed_source_distribution ) ; % PSk ( H | E , S = Sk )
        source_models ( : , : , k ) = model ;
    end
        
    % Learn the domain discriminator model
    
    if uniform
        assumed_sources_priors = ones ( [ 1 , 1 , num_S ] ) / num_S ;
    else
        assumed_sources_priors = rand ( [ 1 , 1 , num_S ] ) ;
        assumed_sources_priors = assumed_sources_priors ./ sum ( assumed_sources_priors ) ; % P' ( S ) = kappa
    end
    assumed_distribution_EHS = distribution_EH_given_S .* assumed_sources_priors ; % P' ( E , H , S )
    assumed_distribution_ES = marginalize_H ( assumed_distribution_EHS ) ; % P' ( E , S )
    domain_discriminator_model = learn_domain_discriminator_model ( assumed_distribution_ES ) ; % P' ( S | E )
    
end
