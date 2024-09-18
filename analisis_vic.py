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
        #return mne.filter.filter_data(X, l_freq=self.fmin, h_freq=self.fmax, sfreq=epochs.info['sfreq'])
        return mne.filter.filter_data(X, l_freq=self.fmin, h_freq=self.fmax, sfreq=self.sfreq)
        #return bandpass(X, self.fmin, self.fmax, self.sfreq)


def train_pfbcsp3(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    # Obtener datos y etiquetas
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        ('csp', csp)
    ])
    
    """
    print("X: ", X_train.shape)
    X_trans=combined_features.transform(X_train)
    print("X_train: ", X_trans.shape)
    csp.fit(X_trans, y_train)
    X_csp=csp.transform(X_trans)
    print("X_csp: ", X_csp.shape)
    """
    
    pipe.fit(X_train, y_train)
    X_csp=pipe.transform(X_train)
    print("X_csp: ", X_csp.shape)
    

def train_pfbcsp2(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    # Obtener datos y etiquetas
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    
    param_grid = {}
    # Definir el espacio de parámetros para GridSearchCV
    param_grid = {}
    
    # Definir RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Crear GridSearchCV
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1)
    # Ajustar GridSearchCV al conjunto de entrenamiento
    results=grid.fit(X_train, y_train)
    
    # Predecir y evaluar el modelo
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Mean Accuracy: %.3f" % results.best_score_)    
    print(f'Best parameters: {grid.best_params_}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
def train_pfbcsp2_notest(file):
    
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

def train_pfbcsp(file):
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
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Crear lista para almacenar las características de cada banda de frecuencia
    X_train_filtered = []
    X_test_filtered = []
    
    # Aplicar filtrado y CSP para cada banda de frecuencia
    for band in frequency_bands:
        fmin, fmax = band
        print("fmin: ", fmin, " - fmax: ", fmax)
        
        # Filtrar épocas en la banda de frecuencia actual
        epochs_band_train=bandpass(X_train, fmin, fmax, sfreq)
        epochs_band_test=bandpass(X_test, fmin, fmax, sfreq)
        
        # Aplicar CSP
        csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
        csp.fit(epochs_band_train, y_train)
        
        # Transformar los datos
        X_train_band = csp.transform(epochs_band_train)
        X_test_band = csp.transform(epochs_band_test)
        
        # Agregar las características a las listas
        X_train_filtered.append(X_train_band)
        X_test_filtered.append(X_test_band)
    
    # Combinar las características de todas las bandas de frecuencia
    X_train_combined = np.concatenate(X_train_filtered, axis=1)
    X_test_combined = np.concatenate(X_test_filtered, axis=1)

    print("X_train_combined: ", X_train_combined.shape)
    print("X_test_combined: ", X_test_combined.shape)
    
    lda=LinearDiscriminantAnalysis()
    #clf = Pipeline([('CSP', csp), ('LDA', lda)])
    
    model = Pipeline([('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    param_grid={}
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X_train_combined, y_train)
    
    print("Mean Accuracy: %.3f" % results.best_score_)
    #df = getResult(results, columns)
    # Predecir y evaluar el modelo
    y_pred = search.predict(X_test_combined)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy csp: {accuracy * 100:.2f}%')



def train_pfbcsp_notest(file):
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
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']
    
    X_train = X
    y_train = y
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Crear lista para almacenar las características de cada banda de frecuencia
    X_train_filtered = []
    
    # Aplicar filtrado y CSP para cada banda de frecuencia
    for band in frequency_bands:
        fmin, fmax = band
        print("fmin: ", fmin, " - fmax: ", fmax)
        
        # Filtrar épocas en la banda de frecuencia actual
        epochs_band_train=bandpass(X_train, fmin, fmax, sfreq)
        
        # Aplicar CSP
        csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
        csp.fit(epochs_band_train, y_train)
        
        # Transformar los datos
        X_train_band = csp.transform(epochs_band_train)
        
        # Agregar las características a las listas
        X_train_filtered.append(X_train_band)
    
    # Combinar las características de todas las bandas de frecuencia
    X_train_combined = np.concatenate(X_train_filtered, axis=1)

    print("X_train_combined: ", X_train_combined.shape)

    lda=LinearDiscriminantAnalysis()
    #clf = Pipeline([('CSP', csp), ('LDA', lda)])
    
    model = Pipeline([('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    param_grid={}
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X_train_combined, y_train)
    
    print("Mean pfbcsp Accuracy: %.3f" % results.best_score_)
    #df = getResult(results, columns)
    # Predecir y evaluar el modelo
    
def train_csp(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    #epochs.filter(8, 15)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train=bandpass(X_train, 8, 30, sfreq)
    X_test=bandpass(X_test, 8, 30, sfreq)

    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda=LinearDiscriminantAnalysis()
    
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    csp.fit(X_train, y_train)
    
    
    # Transformar los datos
    X_train_band = csp.transform(X_train)
    X_test_band = csp.transform(X_test)
    
    
    print("X_train_band: ", X_train_band.shape)
    print("X_test_band: ", X_test_band.shape)
    #clf = Pipeline([('CSP', csp), ('LDA', lda)])
    #lda.fit(X_train_band, y_train)
    
    model = Pipeline([('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    param_grid={}
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X_train_band, y_train)
    
    print("Mean Accuracy: %.3f" % results.best_score_)
    #df = getResult(results, columns)
    # Predecir y evaluar el modelo
    y_pred = search.predict(X_test_band)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy csp: {accuracy * 100:.2f}%')
    
    
def train_csp_notest(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    #epochs.filter(8, 15)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #X=bandpass(X, 8, 30, sfreq)
    X=mne.filter.filter_data(X, l_freq=8, h_freq=30, sfreq=sfreq, verbose="error")

    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda=LinearDiscriminantAnalysis()
    
    print("X_train: ", X.shape)
    print("y_train: ", y.shape)
    csp.fit(X, y)
    
    
    # Transformar los datos
    X_train_band = csp.transform(X)
    
    
    print("X_train_band: ", X_train_band.shape)
    #clf = Pipeline([('CSP', csp), ('LDA', lda)])
    #lda.fit(X_train_band, y_train)
    
    model = Pipeline([('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

    param_grid={}
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X_train_band, y)
    
    print("Mean csp Accuracy: %.3f" % results.best_score_)
    #print("Mean Accuracy: ", results.cv_results_)
    #df = getResult(results, columns)
    
def main():
    logging.getLogger('mne').setLevel(logging.ERROR)
    filename="D3S02I_S01_FI2S15CH"
    file="Epoch/DATA3/" + filename + "-epo.fif"

    train_csp_notest(file)
    
    #train_pfbcsp(file)
    #train_pfbcsp_notest(file)
    
    #train_pfbcsp2(file)
    #train_pfbcsp2_notest(file)
    
    #train_pfbcsp3(file)

if __name__ == "__main__":
    main()