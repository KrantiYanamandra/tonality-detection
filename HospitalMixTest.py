#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:25:43 2018

@author: kranthiyanamandra
"""

import TonalityDetector as TD
import numpy as np


scoreArray = np.zeros(12)  

with np.load('Hospital/HospitalRecords_TestData.npz') as data:
    
    spliceTimes = data['arr_0']
    y = data['arr_2']
    sr = data['arr_3']

segments = np.zeros((len(spliceTimes),20 * sr + 1))

for i in range(len(spliceTimes)):
    
    segments[i,:] = y[int((spliceTimes[i]/1000) * sr - (10 * sr)) : int((spliceTimes[i]/1000) * sr + (10 * sr)) + 1]

for i in range(len(segments)):
    
    scoreArray[i] = TD.TonalityDetector.tonalityDetection(segments[i], sr, True, True, False)
    
np.savetxt('testHospital.csv', scoreArray, delimiter = ',')