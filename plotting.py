#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

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

def main():   
    #low_freq, high_freq = 7., 30.
    
    #raw = mne.io.read_raw_fif("data_eeg.fif", preload=True)
    #events_from_file = mne.read_events("event.fif",)
    data = "D5"
    sujeto = "SB"
    sesion = "S01"
    tipo = "M"
    run = "R1"
    #path="../EEG data/Raw fif/" + sujeto +"/"
    path="Raw/DATA5/" + sujeto + "/" + sesion + "/" 
    
    
    files = ["Raw/DATA3/S02/S01/D3S02I_S01R1_eeg.fif",
             "Raw/DATA4/SB/S02/D4SBI_S02R1_eeg.fif"]
    
    """
    files2 = ["Raw/DATA4/SA/S01/D4SAI_S01R1_eeg.fif",
             "Raw/DATA3/S02/S01/D3S02I_S01R1_eeg.fif"]
    """
    #fileName = sujeto + "_FILT_S1R" + run
    #fileName =  data + sujeto + tipo + "_" + sesion + run    
    #fileRaw = fileName + "_eeg.fif"
    #fileEve = fileName + "-eve.fif"
    
    for file in files: 
        raw_crude = mne.io.read_raw_fif(file, preload=True)
        
        print(raw_crude.info)
        #event, event_ids=mne.read_events(raw_crude)
        #event, event_id=mne.events_from_annotations(raw_crude)
        #events_from_file = mne.read_events(path + fileEve)
        #print(event)
        #print(event_id)    
        #raw_crude.plot(scalings=None, n_channels=8)    
        
        raw = raw_crude.copy().filter(l_freq=1, h_freq=40)    
        raw.plot(scalings=None, n_channels=8)
    
if __name__ == "__main__":
    main()