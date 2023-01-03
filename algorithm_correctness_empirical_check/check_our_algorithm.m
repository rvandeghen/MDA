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
% This function checks empirically the correctness the algorithm on
% 10,000 random cases. If everything goes well, it should answer
% 'The algorithm seems to behave as an exact model.' after a few seconds.
% It is not a formal proof of the correctness of the algorithm (note that
% such a proof is provided in Section 3.3 of the paper). It just means
% that the posteriors obtained by the algorithm for the target domain
% correspond (up to a small tolerance, necessary for numerical reasons)
% to the true posteriors in the target domain, in the 10,000 cases.

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ answer ] = check_our_algorithm ()

    for trial = 1 : 10000
        
        % Randomy choose a problem
        
        [ distribution_EH_given_S , true_sources_priors , true_target_posteriors ] = choose_random_case () ;
        
        % Perform the off-line step.

        [ true_hypotheses_priors , assumed_hypothesis_priors , source_models , assumed_sources_priors , domain_discriminator_model ] = our_algorithm_off_line_step ( distribution_EH_given_S ) ;

        % Perform the on-the-fly step.

        computed_target_posteriors = our_algorithm_on_the_fly_step ( true_hypotheses_priors , assumed_hypothesis_priors , source_models , true_sources_priors , assumed_sources_priors , domain_discriminator_model ) ;

        % Check the result
        
        abs_error = abs ( true_target_posteriors - computed_target_posteriors ) ;
        if abs_error > 1e-4
            abs_error
            answer = 'The algorithm does not behave as an exact model!' ;
            return
        end
        
    end
    answer = 'The algorithm seems to behave as an exact model.' ;
    
end
