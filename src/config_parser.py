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

class Config:
    def __init__(self, dict):
        self.datasets = dict['datasets']
        self.name = dict['name']
        self.num_classes = int(dict['num_classes'])
        self.max_size = {a: int(x) for a, x in dict['max_size'].items()}
        self.substract_background = bool(dict['substract_background'])
        self.img_size = tuple(map(int, dict['img_size'].strip('()').split(',')))
        

    
        