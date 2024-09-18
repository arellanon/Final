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
import pandas as pd

#sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *
import logging


class FrequencyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fmin, fmax, sfreq):
        self.fmin = fmin
        self.fmax = fmax
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return mne.filter.filter_data(X, l_freq=self.fmin, h_freq=self.fmax, sfreq=self.sfreq, verbose="error")
        #return bandpass(X, self.fmin, self.fmax, self.sfreq)


class TimeFrequencyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fmin, fmax, tmin, tmax, sfreq, times):
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.times = times

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_filtered = mne.filter.filter_data(X, l_freq=self.fmin, h_freq=self.fmax, sfreq=self.sfreq)
        time_mask = (self.times  >= self.tmin) & (self.times  <= self.tmax)
        
        # Seleccionar la ventana de tiempo
        #time_mask = (self.times >= self.tmin) & (self.times <= self.tmax)
        #print(time_mask)
        X_windowed = X_filtered[:, :, self.tmin : self.tmax]
        
        """
        # Asegurar que todas las ventanas de tiempo tengan la misma longitud
        if X_windowed.shape[2] != len(self.times[time_mask]):
            diff = len(self.times[time_mask]) - X_windowed.shape[2]
            if diff > 0:
                X_windowed = np.pad(X_windowed, ((0, 0), (0, 0), (0, diff)), mode='constant')
            else:
                X_windowed = X_windowed[:, :, :diff]
        """
        return X_windowed


def train_ptfbcsp(file):
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    # Obtener datos y etiquetas
    X = epochs.get_data()
    
    X = X[:, :, :500]
    print("X: ", X.shape)
    y = epochs.events[:, -1]
    times = epochs.times[:500]

    print("times: ",times.shape)
    #print("times: ",times)
    # Dividir datos en conjuntos de entrenamiento y prueba
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sfreq=epochs.info['sfreq']
    print(epochs.info['sfreq'])
    
    # Definir ventanas de tiempo
    time_windows = [
        (0, 250),
        (63, 313),
        (125, 375),
        (188, 438),
        (250, 500)
    ]
    """
    time_windows = [
        (0.0, 2.0),
        (0.5, 2.5),
        (1.0, 3.0),
        (1.5, 3.5),
        (2.0, 4.0)
    ]
    """
        
    frequency_bands = [
        (4,8),
        (6,10),
        (8,12),
        (10,14),
        (12,16),
        (14,18),
        (16,20),
        (18,22),
        (20,24),
        (22,26),
        (24,28),
        (26,30),
        (28,32),
        (30,34),
        (32,36),
        (34,38),
        (36,40)
    ]
    
    # Crear transformadores de tiempo-frecuencia
    time_freq_filters = [(f'band_{i}_time_{j}', TimeFrequencyFilter(fmin, fmax, tmin, tmax, sfreq, times))
                         for i, (fmin, fmax) in enumerate(frequency_bands)
                         for j, (tmin, tmax) in enumerate(time_windows)]
    
    #print(time_freq_filters[0])
    
    # Crear FeatureUnion para combinar las características de todas las bandas y ventanas de tiempo
    combined_features_tf = FeatureUnion(time_freq_filters)
    
    X_transf=combined_features_tf.transform(X)
    print(X_transf.shape)
    
    # Definir el pipeline
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda=LinearDiscriminantAnalysis()
    
    pipe = Pipeline([
        ('filter', combined_features_tf),
        ('csp', csp),
        ('lda', lda)
    ])
    

    # Definir el espacio de parámetros para GridSearchCV
    param_grid = {}
    
    # Definir RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    
    # Crear GridSearchCV
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
    # Ajustar GridSearchCV al conjunto de entrenamiento
    results=grid.fit(X, y)
    
    # Predecir y evaluar el modelo
    #y_pred = grid.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    
    print("Mean Accuracy: %.3f" % results.best_score_)    
    #print(f'Best parameters: {grid.best_params_}')
    #print(f'Accuracy: {accuracy * 100:.2f}%')

def train_pfbcsp(file):
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    # Obtener datos y etiquetas
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sfreq=epochs.info['sfreq']
    print(epochs.info['sfreq'])
        
    frequency_bands = [
        (4,8),
        (6,10),
        (8,12),
        (10,14),
        (12,16),
        (14,18),
        (16,20),
        (18,22),
        (20,24),
        (22,26),
        (24,28),
        (26,30),
        (28,32),
        (30,34),
        (32,36),
        (34,38),
        (36,40)
    ]
    
    # Crear transformadores de frecuencia para cada banda
    filters = [(f'band_{i}', FrequencyFilter(fmin, fmax, sfreq)) for i, (fmin, fmax) in enumerate(frequency_bands)]
    # Crear FeatureUnion para combinar las características de todas las bandas
    combined_features = FeatureUnion(filters)
    
    # Definir el pipeline
    #csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    #svm = SVC(kernel='linear')    
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda=LinearDiscriminantAnalysis()
    
    pipe = Pipeline([
        ('filter', combined_features),
        ('csp', csp),
        ('lda', lda)
    ])
    

    # Definir el espacio de parámetros para GridSearchCV
    param_grid = {}
    
    # Definir RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    
    # Crear GridSearchCV
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
    # Ajustar GridSearchCV al conjunto de entrenamiento
    results=grid.fit(X, y)
    
    # Predecir y evaluar el modelo
    #y_pred = grid.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    
    print("Mean Accuracy: %.3f" % results.best_score_)    
    #print(f'Best parameters: {grid.best_params_}')
    #print(f'Accuracy: {accuracy * 100:.2f}%')

    
def train_csp(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose="error")
    #epochs.filter(8, 15)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #X_filt=bandpass(X, 8, 30, sfreq)
    X_filt=mne.filter.filter_data(X, l_freq=8, h_freq=30, sfreq=sfreq, verbose="error")

    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda=LinearDiscriminantAnalysis()
    
    print("X_train: ", X.shape)
    print("y_train: ", y.shape)
    #csp.fit(X, y)
    # Transformar los datos
    #X_train_band = csp.transform(X)
    
    
    #print("X_train_band: ", X_train_band.shape)
    
    #lda.fit(X_train_band, y_train)
    
    model= Pipeline([('CSP', csp), ('LDA', lda)])
    
    #model = Pipeline([('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    param_grid={}
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    results = search.fit(X_filt, y)
    
    print("Mean csp Accuracy: %.3f" % results.best_score_)
    #print("Mean Accuracy: ", results.cv_results_)
    #df = getResult(results, columns)
    
    
def main():    
    logging.getLogger('mne').setLevel(logging.ERROR)
    filename="D3S02I_S01_FI4S15CH"
    file="Epoch/DATA3/" + filename + "-epo.fif"

    #train_csp(file)
    train_pfbcsp(file)
    #train_ptfbcsp(file)
    
if __name__ == "__main__":
    main()