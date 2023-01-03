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
% This function computes the exact domain discriminator model for any given
% join distribution of evidences and source domains. This model gives the
% distribution of source domains for each possible evidence.
% 
% If the join distribution of evidences and source domains given in
% argument is shifted, then the model is "exact up to a target shift".

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ model_sources_posteriors ] = learn_domain_discriminator_model ( distribution )
    % P' ( S | E ) = P' ( E , S ) / P' ( E )
    model_sources_posteriors = distribution ./ marginalize_S ( distribution ) ;
end
