#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:59:47 2020

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
#from loaddata import *

#sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import joblib

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *


class GenerateEpoch:

    def run(self):
        #tmin, tmax = 0.5, 2.5   #Rango para DATA1
        tmin, tmax = 0, 4      #Rango para DATA3          
        print("tmin: ",tmin," - tmax: ", tmax )

        #list_channel = ['Pz','Cz','T4','P4','C4','T3','P3','C3']  #TT2 #RR2
        list_channel = ['Pz','Cz','T6','P4','C4','T5','P3','C3']   #TT3 #RR3
        fil_filter = True
        fil_channel = False

        #dataSets = ["DATA1", "DATA3"]
        dataSets = ["DATA4"]
        
        for dataSet in dataSets:
        
            prefijo, tipo, sujetos = getSujetos(dataSet)
            
            for sujeto in sujetos:
                sesiones = getSessiones(dataSet, sujeto)
                
                for sesion in sesiones:
                
                    pathRaw = "Raw/"+ dataSet +"/" + sujeto + "/" + sesion + "/"
                    pathEpoch = "Epoch/"+ dataSet +"/"
                    fileNameOut = prefijo + sujeto + tipo + "_" +sesion
                    files = getFiles(dataSet, sujeto, tipo, sesion)
                    epochs = []
                        
                    for file in files:
                        raw_crude = mne.io.read_raw_fif(pathRaw + file + "_eeg.fif", preload=True)
                        events, event_id=mne.events_from_annotations(raw_crude)
            
                        if fil_channel :
                            raw_crude = raw_crude.pick_channels(list_channel)
                        
                        if fil_filter :
                            param_filter = dict(order=5, ftype='butter') 
                            raw = raw_crude.copy().filter(l_freq=1, h_freq=40, method= "iir", iir_params = param_filter)
                        else:
                            raw = raw_crude
                        
                        #Se genera las epocas con los datos crudos y los eventos
                        epoch = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
                        epochs.append(epoch)
                    epoch_final=mne.concatenate_epochs(epochs, add_offset=True, on_mismatch='raise', verbose=None)
                    print(epoch_final)
                    epoch_final.save(pathEpoch  + fileNameOut + "-epo.fif", overwrite=True)
                
                
def getSujetos(dataSet):
    if dataSet == "DATA1":
        sujetos =  ["SX"]
        return "D1", "I", sujetos 
    if dataSet == "DATA3": 
        sujetos = ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S12"]
        return "D3", "I", sujetos
    if dataSet == "DATA4": 
        sujetos = ["SA", "SB"]
        return "D4", "I", sujetos
    if dataSet == "DATA5":
        sujetos = ["SA", "SB"]
        return "D5", "M", sujetos

def getSessiones(dataSet, sujeto):
    if dataSet == "DATA1":
        sesiones =  ["S01"]
        return sesiones 
    if dataSet == "DATA3":
        sesiones =  ["S01"]
        return sesiones
    if dataSet == "DATA4":
        if sujeto == "SA":
            sesiones =  ["S01", "S02", "S03"]
            return sesiones
        if sujeto == "SB":
            sesiones =  ["S02"]
            return sesiones
    if dataSet == "DATA5":
        if sujeto == "SA":
            sesiones =  ["S01", "S02", "S03"]
            return sesiones
        if sujeto == "SB":
            sesiones =  ["S01", "S02"]
            return sesiones    

def getFiles(dataSet, sujeto, tipo, sesion):
    if dataSet == "DATA1":
        filename = "D1" + sujeto + tipo + "_" + sesion + "R1"
        files =  [filename]
        return files 
    if dataSet == "DATA3": 
        runs = ["1", "2", "3", "4"]
        files = []
        for run in runs:
            files.append("D3" + sujeto + tipo + "_" + sesion + "R" + run)
        return files 
    if dataSet == "DATA4": 
        if sujeto == "SA":
            runs = ["1", "2", "3", "4"]
            files = []
            for run in runs:
                files.append("D4" + sujeto + tipo + "_" + sesion + "R" + run)
            return files
        if sujeto == "SB":
            runs = ["1", "2", "3", "4"]
            files = []
            for run in runs:
                files.append("D4" + sujeto + tipo + "_" + sesion + "R" + run)
            return files
    if dataSet == "DATA5": 
        runs = ["1", "2", "3", "4"]
        files = []
        for run in runs:
            files.append("D5" + sujeto + tipo + "_" + sesion + "R" + run)
        return files     


def main():
    print("Inicio...")
    generate = GenerateEpoch()
    generate.run()
    print("Fin...")
    


if __name__ == "__main__":
    main()