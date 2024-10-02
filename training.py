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
from sklearn.neighbors import KNeighborsClassifier

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

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from libb import *
import logging

import warnings
warnings.filterwarnings("ignore")


# Tipos de extraccion de caracteristicas
def runCSP(file):
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

def runPFBCSP(file):
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

def runPTFBCSP(file):
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


# Configuracion de parametros para los modelos.
def getParamLDA():
    param_solver1 = {
        "LDA__solver": ["svd"],
        'LDA__store_covariance': [True, False],
        'LDA__n_components': [None, 1],
        "LDA__tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
    
    param_solver2 = {
        "LDA__solver": ["lsqr", "eigen"],
        'LDA__n_components': [None, 1],
        "LDA__shrinkage": [None, 'auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }    
    param_grid = [param_solver1, param_solver2]
    return param_grid

def getParamSVM():
    param_grid = {
        'SVM__C': [0.01, 0.1, 1, 10, 100],
        'SVM__kernel': ['linear', 'rbf'],
        'SVM__gamma': [0.01, 0.1, 1],
        'SVM__class_weight': [None, 'balanced']
    }
    return param_grid

def getParamKNN():
    param_grid = {
        'KNN__n_neighbors': [3, 5, 7, 9, 12],
        'KNN__weights': ['uniform', 'distance'],
        'KNN__metric': ['euclidean', 'manhattan', 'minkowski'],
        'KNN__algorithm': ['auto', 'ball_tree']
    }
    return param_grid

def getParamANN():
    param_grid = {
        'ANN__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'ANN__activation': ['relu', 'tanh'],
        'ANN__solver': ['adam', 'sgd'],
        'ANN__learning_rate': ['constant', 'adaptive'],
        'ANN__alpha': [0.0001, 0.001],
        'ANN__max_iter': [200]
    }
    return param_grid

# Retorna dataframe
def getResult(results, columns, file,  method, classification, seconds_total):
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    logging.info("Mean Accuracy: %.3f" % results.best_score_)
    logging.info("Config: %s" % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[columns]
    
    #df = pd.DataFrame(results.cv_results_)
    #new_column = file
    df.insert(loc=0, column='File', value=file)
    df.insert(loc=1, column='method', value=method)
    df.insert(loc=2, column='classification', value=classification)
    df.insert(loc=3, column='seconds_total', value=seconds_total)
    
    
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(seconds_total))
    print("Tiempo de ejecucion: ", formatted_time, "- segundos: ", seconds_total)
    #logging.info("Tiempo de ejecucion: %.3f" % tduracion)
    logging.info("Tiempo de ejecución: %s  - segundos:  %.3f" % (formatted_time, seconds_total))
    
    return df

# Ejecucion de extraccion de caracteristicas
def runFeatureExtraction(method, file):
    if method =="CSP" :
        X, y = runCSP(file)
    elif method == "PFBCSP" :
        X, y = runPFBCSP(file)
    elif method == "PTFBCSP" :
        X, y = runPTFBCSP(file)
    return X, y

# Devuelve modelo de clasificacion y sus parametros
def getModel(clf):
    if clf == "LDA" :
        lda=LinearDiscriminantAnalysis()
        model = Pipeline([('LDA', lda)])
        param_grid = getParamLDA()
    elif clf == "SVM" :
        svm = SVC(kernel='linear') # Linear Kernel
        model = Pipeline([('SVM', svm)])
        param_grid = getParamSVM()
    elif clf == "KNN" :
        knn = KNeighborsClassifier(n_neighbors=3)
        model = Pipeline([('KNN', knn)])
        param_grid = getParamKNN()
    elif clf == "ANN" :
        ann = MLPClassifier(random_state=1, max_iter=200)
        model = Pipeline([('ANN', ann)])
        param_grid = getParamANN()
    return model, param_grid

def runProcess(file, method, classification):
    #Tiempo de inicio
    inicio = time.time()
    print("\nFILE: ", file)
    print("METODO DE EXTRACION DE CARACTERISTICAS: ", method)
    print("MODELO DE CLASIFICACION: ", classification)
    
    logging.info("--------------------------------------------------------------------------------------------------------")
    logging.info("FILE: %s" % file)
    logging.info("METODO DE EXTRACION DE CARACTERISTICAS: %s" % method)
    logging.info("MODELO DE CLASIFICACION: %s" % classification)
    
    #Se realiza la extracion de caracteristicas
    X, y = runFeatureExtraction(method, file)
    
    print("Tranformacion X: ", X.shape)
    print("Tranformacion y: ", y.shape)
    
    logging.info("X: %s" % str(X.shape) )
    logging.info("y: %s" % str(y.shape) )
        
    #Obtiene el modelo de clasificacion y sus parametros
    model, param_grid = getModel(classification)
        
    #Cross-Validation 10x10
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    
    #Metricas
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': 'roc_auc',  # Agregar ROC AUC,
        'log_loss': 'neg_log_loss'  # Agregar Log Loss
    }
    
    #Busca la mejor configuracion
    search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X, y)

    #columns =[ "params", "mean_test_score", "std_test_score"]
    columns =["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params", "mean_test_accuracy", "std_test_accuracy", "mean_test_precision", "std_test_precision", "mean_test_recall", "std_test_recall", "mean_test_f1", "std_test_f1", "mean_test_roc_auc", "std_test_roc_auc", "mean_test_log_loss", "std_test_log_loss"]
    #columns =["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params", "mean_test_accuracy", "std_test_accuracy", "mean_test_recall", "std_test_recall", "mean_test_f1", "std_test_f1"]    

    
    #Se guarda resultados y modelo
    #joblib.dump(search, path + filename + ".pkl")
    #Tiempo final
    fin = time.time()
    tduracion = fin-inicio
    
    df = getResult(results, columns, file, method, classification, tduracion)
    return df

    
def run(folder_input, folder_output, dataset):
    logging.info('input: %s ', folder_input)
    logging.info('output: %s ', folder_output)
    
    #Definimos las extracciones de caracteristicas a analizar
    method_feature_extraction = ["CSP", "PFBCSP", "PTFBCSP"]
    #method_feature_extraction = ["CSP"]
    
    #Definimos los modelos de clasificacion a analizar
    classifications = ["LDA", "SVM", "KNN", "ANN"]
    #classifications = ["LDA"]
    
    first_total = True
    fout_total = folder_output + dataset + ".csv"
    
    for method in method_feature_extraction:
        
        first_method = True
        fout_method = folder_output + method + ".csv"
        for clf in classifications:
            first_clf = True
            fout_clf = folder_output + method + "_" + clf + ".csv"
            for root, dirs, files in os.walk(folder_input):
                for file in files:
                    filename=os.path.join(root, file)
                    df = runProcess(filename, method, clf)
                    if first_clf: 
                        df_clf = df
                    else :
                        df_clf = pd.concat([df_clf, df], ignore_index=True)    
                    first_clf = False
            
            #Se guarda archivo por metodo y clasificador
            df_clf.to_csv(fout_clf, mode="w", index=False, header=True)
            
            if first_method:
                df_method = df_clf
            else :
                df_method = pd.concat([df_method, df_clf], ignore_index=True)
            first_method = False
        
        #Se guarda archivo por metodo y clasificador    
        df_method.to_csv(fout_method, mode="w", index=False, header=True)        
        if first_total:
            df_total = df_method
        else :
            df_total = pd.concat([df_total, df_method], ignore_index=True)
        first_total = False    
    #Se guarda archivo de resultados del dataset
    df_total.to_csv(fout_total, mode="w", index=False, header=True)
                             
def main():
    logging.getLogger('mne').setLevel(logging.ERROR)
    logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG, format='%(asctime)s %(message)s')
    # Suprimir la advertencia de convergencia    
    dataset = "DATA3"
    #dataset = "DATAQ"
    folder_input = "Epoch/" + dataset + "/"
    folder_output= "Output6/" + dataset + "/"
    
    total_inicio=time.time()
    #EJECUCION DEL PROGRAMA
    run(folder_input, folder_output, dataset)
    
    total_final= time.time()
    total_duracion = total_final-total_inicio
    
    total_time = time.strftime("%H:%M:%S", time.gmtime(total_duracion))
    print("Tiempo total de ejecucion: ", total_time, "- segundos: ", total_duracion)
    #logging.info("Tiempo de ejecucion: %.3f" % tduracion)
    logging.info("Tiempo total de ejecución: %s  - segundos:  %.3f" % (total_time, total_duracion))

if __name__ == "__main__":
    main()