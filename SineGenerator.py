#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:03:23 2018

@author: kranthiyanamandra
"""

import numpy as np


def sineGenerator(frequency):
    
    
    sr = 22050
    duration = 15
    
    samples = (np.sin(2 * np.pi * np.arange (sr * duration) * frequency / sr)).astype(np.float32)
    
    return samples
    
        
        
