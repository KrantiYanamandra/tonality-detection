#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:09:57 2018

@author: kranthiyanamandra
"""

import TonalityDetector as TD
import numpy as np
import os
import glob

dirName = os.path.dirname(__file__)
pathAllAudioData = dirName + "/Mixes"
pathRangeAudio = glob.glob(pathAllAudioData + '/**/*.mp3', recursive=True)

#tonalityScore = np.zeros([604, ])

scoreArray = np.zeros(604)

final = [pathRangeAudio,np.zeros(604)]

    
for index, value in enumerate(pathRangeAudio):
   
   scoreArray[index] = TD.TonalityDetector.tonalityDetection(True, True, False)
   
   final[0][index] = pathRangeAudio[index]
   final[1][index] = scoreArray[index]

final = np.asarray(final).T
np.savetxt('test.csv', final, delimiter = ',', fmt = '%s')
   
   
    
   

