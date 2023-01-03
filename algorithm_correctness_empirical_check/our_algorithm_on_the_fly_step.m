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
% This function provides an easy to read implementation of the "on-the-fly
% step" of the unsupervised domain adaptation algorithm presented in
% Section 3.4 of the paper, for the mixture domain adaptation problem
% specified in Section 3.2.
%
% See Figure 4 in the paper.

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ target_posteriors ] = our_algorithm_on_the_fly_step ( true_hypotheses_priors , assumed_hypothesis_priors , source_models , true_sources_priors , assumed_sources_priors , domain_discriminator_model )
    
    num_E = size ( source_models , 1 ) ;
    num_H = size ( source_models , 2 ) ;
    num_S = size ( source_models , 3 ) ;

    % Apply a target shift on the output of the source models
    
    for k = 1 : num_S
        model = source_models ( : , : , k ) ; % PSk ( H | E , S = Sk )
        true_priors = true_hypotheses_priors ( : , : , k ) ;
        assumed_priors = assumed_hypothesis_priors ( : , : , k ) ;
        model = model .* true_priors ./ assumed_priors ;
        model = model ./ sum ( model , 2 ) ; % P ( H | E , S = Sk )
        source_models ( : , : , k ) = model ;
    end
    
    % Apply a target shift on the output of the domain discriminator model
    
    model = domain_discriminator_model ; % P' ( S | E )
    model = model .* true_sources_priors ./ assumed_sources_priors ;
    model = model ./ sum ( model , 3 ) ; % P ( S | E )
    domain_discriminator_model = model ;
    
    % Apply the combination
    
    tmp = source_models .* domain_discriminator_model ; % P ( H , S | E ) = P ( H | E , S ) * P ( S | E )
    target_posteriors = marginalize_S ( tmp ) ; % P ( H | E )

end
