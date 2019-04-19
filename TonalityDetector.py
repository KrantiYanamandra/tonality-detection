#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:20:02 2018

@author: kranthiyanamandra
"""

import matplotlib.pyplot as plt
import numpy as np
import madmom
#import librosa
import operator
import yam
import pyloudnorm


class TonalityDetector:
    
    @staticmethod
    def tonalityDetection(y, sr, specWhiten = False, loudNorm = False, plots = False):
        
        # Load the audio
        #y, sr = librosa.load(filePath, sr = None)
        
        # Take the specified length of audio
        #y = y[0 : length * sr]
        
        if specWhiten == True:
            
            # Apply spectral whitening
            y = yam.correlate.spectral_whitening(data = y, sr = sr, smooth = 256, filter = [10, 10000])
        
        if loudNorm == True:
            
            # Create loudness meter and calculate loudness
            meter = pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(y)
            
            # Loudness normalise the audio to -6 dB
            y = pyloudnorm.normalize.loudness(y, loudness, -6)
        
        # Deep chroma processing
        dcp = madmom.audio.DeepChromaProcessor()
        
        chroma_ = dcp(y)
        
        chroma = chroma_.T
        
        # Summing all chroma energies across time
        chromaEnergy = chroma.sum(axis = 0)
        
        # Normalising chroma energies
        chromaNormalised = chromaEnergy / max(chromaEnergy)
        
        # Calculate duration of audio
        duration = len(y) / sr
        
        # Calculate tonality score as sum of normalised chroma over the duration of the audio
        tonality = np.sum(chromaNormalised) / duration
        
        # Calculate the amount of energy of each pitch class
        pitchClassEnergy = chroma.sum(axis = 1)
        rootNoteEnergy = max(pitchClassEnergy)
        
        # Full spectrum reference chroma calculations
#        referenceChroma = np.ones((12, chroma.shape[1]))
#        refChromaEnergy = referenceChroma.sum(axis=0)
#        refChromaNorm = refChromaEnergy / max(refChromaEnergy)
#        refTonality = np.sum(refChromaNorm) / duration
        
        # Scaling the tonality value to lie between 0 and 1
        scaledTonality = TonalityDetector.scaleValue(tonality, 0, 5, 0, 0.5)
    
        threshold = 20
        rootNote = None
        if rootNoteEnergy >= threshold:
                
            index, value = max(enumerate(pitchClassEnergy), key = operator.itemgetter(1))
            
            #finding first three indices of pitch classes that have the max energy
            #indices = np.argpartition(pitchClassEnergy, -3)[-3:]
            
            #sort 
            #indices = indices[np.argsort(pitchClassEnergy[indices])]
    
            if index == 0:
                rootNote = 'C'
            
            elif index == 1:
                rootNote = 'C#'
            
            elif index == 2:
                rootNote = 'D'
            
            elif index == 3:
                rootNote = 'D#'
            
            elif index == 4:
                rootNote = 'E'
            
            elif index == 5:
                rootNote = 'F'
            
            elif index == 6:
                rootNote = 'F#'
            
            elif index == 7:
                rootNote = 'G'
            
            elif index == 8:
                rootNote = 'G#'
            
            elif index == 9:
                rootNote = 'A'
            
            elif index == 10:
                rootNote = 'A#'
            
            elif index == 11:
                rootNote = 'B'
            
            elif index == 12:
                rootNote = 'B#'
                
        elif rootNoteEnergy < threshold:
            
            print('You need ear training')
        
        if plots == True:
            
            # Plotting the chromagram and chroma energies
            plt.subplot(2,1,1)
            plt.imshow(chroma_.T, origin = 'lower', aspect = 'auto')
            
            plt.subplot(2,1,2)
            plt.plot(np.arange(0, duration, duration / len(chromaNormalised)), chromaEnergy)
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Chroma energy')
            plt.xticks(np.arange(0, duration, step = 10))
            
            plt.yticks(np.arange(0, 4, step = 0.5))
            
            plt.axis([0, duration, 0, 4])
            plt.show()
            
            plt.tight_layout()
        
        return scaledTonality
        
    
    @staticmethod    
    def scaleValue(value, leftMin, leftMax, rightMin, rightMax):
        
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
    
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
    
        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)


    
    