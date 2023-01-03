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
% This function marginalizes a distribution with respect to the source
% domains.
% 
% The distributions are represented by three-dimentional matrices.
% - The first dimension stands for the the evidences.
% - The second dimension stands for the hypotheses.
% - The third dimension stands for the source domains.
% The elements of these matrices are the probbailities for the
% corresponding evidence, hypothesis and source domain.

% In this file:
%     E stands for "evidence"
%     H stands for "hypothesis"
%     S stands for "source domain"

function [ out ] = marginalize_S ( in )
    out = sum ( in , 3 ) ;
end
