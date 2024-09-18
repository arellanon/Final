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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib

#mne
import mne
from mne.decoding import CSP

from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *
import logging

def getResult(results, columns, file):
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    logging.info("Mean Accuracy: %.3f" % results.best_score_)
    logging.info("Config: %s" % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[
        columns
    ]
    #new_column = file
    df.insert(loc=0, column='File', value=file)
    return df

def getParamLDA():
    param_solver1 = {
        "LDA__solver": ["svd"],
        "LDA__store_covariance": [True, False],
        "LDA__tol": np.array([0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001])
    }
    
    param_solver2 = {
        "LDA__solver": ["lsqr", "eigen"]
        #"LDA__shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
    }
    
    param_solver3 = {
        "LDA__solver": ["lsqr"],
        "LDA__shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
    }
    
    #param_grid = [param_solver1, param_solver2]
    #param_grid = param_solver3
    param_grid = [param_solver1, param_solver3]
    return param_grid

def getParamCSP():
    param_solver1 = {
        "CSP__n_components": [2],
        "CSP__reg": [None],
        "CSP__log": [True, False],
        "CSP__norm_trace": [True, False]
    }
    
    param_grid = param_solver1
    return param_grid

def getParamSVM():
    param_solver1 = {
        #"SVM__kernel": ["poly", "linear", "rbf", "sigmoid", "precomputed"]
        "SVM__kernel": ["poly","linear","rbf","sigmoid"]
    }
    
    param_grid = param_solver1
    return param_grid

def getParamTREE():
    param_solver1 = {
        "TREE__criterion": ["gini", "entropy", "log_loss"]
    }
    
    param_grid = param_solver1
    return param_grid

def getParamRNA():
    param_solver1 = {
        "RNA__solver": ["lbfgs", "sgd", "adam"]
    }
    
    param_grid = param_solver1
    return param_grid


def train_ptfbcsp(file):
    frequency_bands = [
        (4,8),   (6,10),   (8,12), (10,14), (12,16),
        (14,18), (16,20), (18,22), (20,24), (22,26), 
        (24,28), (26,30), (28,32), (30,34), (32,36),
        (34,38), (36,40)
    ]

    time_windows_2s = [
        (0, 125),
        (25, 150),
        (50, 175),
        (75, 200),
        (100, 225),
        (125, 250)
    ]
    
    time_windows_4s = [
        (0, 250),
        (50, 300),
        (100, 350),
        (150, 400),
        (200, 450),
        (250, 500),
    ]
    
    time_windows_250_2s = [
        (0, 500),
        (100, 600),
        (200, 700),
        (300, 800),
        (400, 900),
        (500, 1000),
    ]
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']
    time=epochs.times.shape[0]-1
    print("time: ",time)
    
    if time==500:
        time_windows=time_windows_4s
    elif time==250:
        time_windows=time_windows_2s
    elif time==1000:
        time_windows=time_windows_250_2s        
    
    print(time_windows)
    print("sfreq: ", sfreq)
    
    X_train = X
    y_train = y
    
    print("X: ", X.shape)
    print("y: ", y.shape)
        
    # Crear lista para almacenar las características de cada banda de frecuencia
    X_train_filtered = []
    
    # Aplicar filtrado y CSP para cada banda de frecuencia
    
    for band in frequency_bands:
        fmin, fmax = band
        #print("fmin: ", fmin, " - fmax: ", fmax)        
        # Filtrar épocas en la banda de frecuencia actual
        epochs_band_train=bandpass(X_train, fmin, fmax, sfreq)
        
        for times in time_windows_4s:
            tmin, tmax = times
            #print("tmin: ", tmin, " - tmax: ", tmax)
            epochs_time_train=epochs_band_train[:,:,tmin:tmax]
            
            # Aplicar CSP
            csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
            csp.fit(epochs_time_train, y_train)
            
            # Transformar los datos
            X_train_band = csp.transform(epochs_time_train)
            
            # Agregar las características a las listas
            X_train_filtered.append(X_train_band)

    
    # Combinar las características de todas las bandas de frecuencia
    X_train_combined = np.concatenate(X_train_filtered, axis=1)
    print("X_train_combined: ", X_train_combined.shape)
    return X_train_combined, y


def train_pfbcsp(file):
    frequency_bands = [
        (4,8),   (6,10),   (8,12), (10,14), (12,16),
        (14,18), (16,20), (18,22), (20,24), (22,26), 
        (24,28), (26,30), (28,32), (30,34), (32,36),
        (34,38), (36,40)
    ]
    
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']
    
    X_train = X
    y_train = y
    
    print("X: ", X.shape)
    print("y: ", y.shape)
        
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
    return X_train_combined, y
    
def train_csp(file):
    epochs=mne.read_epochs(file, proj=True, preload=True, verbose=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq=epochs.info['sfreq']
    
    print("X: ", X.shape)
    print("y: ", y.shape)

    #Filtro banda 8 - 30 Hz
    X=bandpass(X, 8, 30, sfreq)
    
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)

    #Ejecuto csp
    csp.fit(X, y)
    # Transformar los datos
    X_transform = csp.transform(X)
    return X_transform, y

def train(file, fil_spatial, clf):
    if fil_spatial =="CSP" :
        X, y = train_csp(file)
    elif fil_spatial == "PFBCSP" :
        X, y = train_pfbcsp(file)
    elif fil_spatial == "PTFBCSP" :
        X, y = train_ptfbcsp(file)
    
    print("Tranformacion X: ", X.shape)
    print("Tranformacion y: ", y.shape)
    
    columns =[ "params", "mean_test_score", "std_test_score"]
    
    if clf == "LDA" :
        lda=LinearDiscriminantAnalysis()
        model = Pipeline([('LDA', lda)])
        param_grid = getParamLDA()
    elif clf == "SVM" :
        svm = SVC(kernel='linear') # Linear Kernel
        model = Pipeline([('SVM', svm)])
        param_grid = getParamSVM()
    elif clf == "TREE" :
        tree = DecisionTreeClassifier(random_state=0)
        model = Pipeline([('TREE', tree)])
        param_grid = getParamTREE()
    elif clf == "RNA" :
        rna = MLPClassifier(random_state=1, max_iter=300)
        model = Pipeline([('RNA', rna)])
        param_grid = getParamRNA()
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    
    #Busca la mejor configuracion
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X, y)
    
    df = getResult(results, columns, file)
    
    #Se guarda resultados y modelo

    #joblib.dump(search, path + filename + ".pkl")
    logging.info('Finished')
    return df
    
    
def run(folder_input, folder_output):
    logging.info('output: %s ', folder_output)
    fil_spatial = ["CSP","PFBCSP", "PTFBCSP"]
    #fil_spatial = ["PTFBCSP"]
    #clfs = ["LDA"]
    clfs = ["LDA", "SVM", "TREE", "RNA"]
    
    for fs in fil_spatial:
        for clf in clfs:
            firts = True
            fout= folder_output + fs + "_" + clf + ".csv"
            for root, dirs, files in os.walk(folder_input):
                for file in files:
                    filename=os.path.join(root, file)    
                    df = train(filename, fs, clf)
                    if firts :
                        df.to_csv(fout, mode="w", index=False, header=True)
                        firts = False
                    else : 
                        df.to_csv(fout, mode="a", index=False, header=False)
    
def main():
    logging.getLogger('mne').setLevel(logging.ERROR)
    #columns_lda=["param_LDA__solver", "param_LDA__shrinkage", "param_LDA__store_covariance", "param_LDA__tol", "params", "mean_test_score", "std_test_score"]
    #columns_csp=["param_CSP__log", "param_CSP__n_components", "param_CSP__norm_trace", "param_CSP__reg", "params", "mean_test_score", "std_test_score"]
    logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG, format='%(asctime)s %(message)s')
    
    dataset = "/DATA4/"
    folder_input = "Epoch" + dataset
    folder_output= "Output" + dataset
    
    #logging.info("LDA")
    param_grid = {}
    columns =[ "params", "mean_test_score", "std_test_score"]
    ML = "CSP"
    
    run(folder_input, folder_output)

if __name__ == "__main__":
    main()