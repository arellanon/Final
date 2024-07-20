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

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *
import logging

def getData(filename):
    epochs=mne.read_epochs(filename, proj=True, preload=True, verbose=None)
    
    #if filter_band: epochs.filter(8, 15)
    #epochs.filter(8, 15)
    data_origen = epochs.get_data()
    
    #print(epochs.events)
    event = epochs.events[:, -1]
    print("data: ", data_origen.shape)
    print("event: ", event.shape)
    
    #data=data_origen
    data=bandpass(data_origen, 8, 15, 250)
    logging.info('Filename: %s ', filename)
    #logging.info('Filter Band (8-15 hz): %s ', filter_band)
    #logging.info('Channel: %s ', channels)
    return data, event

def getResult(results, columns):
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    logging.info("Mean Accuracy: %.3f" % results.best_score_)
    logging.info("Config: %s" % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[
        columns
    ]
    return df


def getParamLDA():
    param_solver1 = {
        "LDA__solver": ["svd"],
        "LDA__store_covariance": [True, False],
        "LDA__tol": np.array([0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001])
    }
    
    param_solver2 = {
        "LDA__solver": ["lsqr", "eigen"],
        "LDA__shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
    }
    
    param_grid = [param_solver1, param_solver2]
    return param_grid


def getParamCSP():
    param_solver1 = {
        "CSP__n_components": [2]
        #"CSP__reg": [None],
        #"CSP__log": [True],
        #"CSP__norm_trace": [True]
    }
    
    param_grid = param_solver1
    return param_grid


def MyCsp(e1, e2):    
    # Train the CSP on the training set only
    W = csp(e1, e2)    
    
    print("W shape:", W.shape)
    #print("W: ", W)
    
    # Apply the CSP on both the training and test set
    e1_mix = apply_mix(W, e1)
    e2_mix = apply_mix(W, e2)

    
    # Select only the first and last components for classification
    comp = np.array([0,-1])
    print("comp: ", comp)
    e1 = e1_mix[:,comp,:]
    e2 = e2_mix[:,comp,:]
    
    # Calculate the log-var
    e1 = logvar(e1)
    e2 = logvar(e2)
    
    return e1, e2


def preparar_data(X, y):
    e1 = X[y==1]
    e2 = X[y==2]
    
    train_percentage = 0.8
    
    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_e1 = int(e1.shape[0] * train_percentage)
    ntrain_e2 = int(e2.shape[0] * train_percentage)
    ntest_e2 = e1.shape[0] - ntrain_e1
    ntest_e1 = e2.shape[0] - ntrain_e2
    
    
    train_e1=e1[:ntrain_e1,:,:]
    train_e2=e2[:ntrain_e2,:,:]    
    
    test_e1=e1[ntrain_e1:,:,:]
    test_e2=e2[ntrain_e2:,:,:]
    
    X_train = np.concatenate((train_e1, train_e2))
    X_test = np.concatenate((test_e1, test_e2))
    
    y_train = np.array([1]*64 + [2]*64)
    y_test = np.array([1]*16 + [2]*16)
    
    return X_train, X_test, y_train, y_test


def train(X, y, path, filename, param_grid, columns):
    logging.info("DATASET: %s" % filename)
    logging.info('Started')
    logging.info("X: %s ", X.shape)
    logging.info("y: %s ", y.shape)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False )
    
    
    X_train, X_test, y_train, y_test = preparar_data(X, y)
    
    cspMNE = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    #lda=LinearDiscriminantAnalysis()
    
    param_grid = {}
    
    e1 = X_train[y_train==1]
    e2 = X_train[y_train==2]
    
    #print("e1", e1)
    
    e1_r, e2_r = MyCsp(e1, e2)
    
    #print("X: ", X.shape)
    #print("X: ", X_train.shape)
    print("e1 ", e1_r)
    
    
    res1 = cspMNE.fit_transform(X_train, y_train)
    
    print("res1 ", res1[y_train==1])
    #print(res1.shape)
    #print(y.shape)
    
    lda=LinearDiscriminantAnalysis()
    #Modelo utiliza CSP y LDA    
    model = Pipeline([('CSP', cspMNE), ('LDA', lda)])
    
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(score)
    result=model.predict(X_test)
    matriz=met.confusion_matrix(y_test, result)
    report=met.classification_report(y_test, result)
    print(matriz)
    print(report)
    
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X, y)
    
    df = getResult(results, columns)
    new_column = filename
    df.insert(loc=0, column='File', value=new_column)
    
    #Se guarda resultados y modelo

    #joblib.dump(search, path + filename + ".pkl")
    logging.info('Finished')
    return df
    """
    
    
def run(folder_input, folder_output, param_grid, columns, ML):
    logging.info('output: %s ', folder_output)
    logging.info('param_grid: %s ', param_grid)
    logging.info('columns: %s ', columns)
    
    firts = True
    file="D4SBI_S02_FI4S8ACH"
    filename="Epoch/DATA4/" + file + "-epo.fif"
    X, y = getData(filename)
    train(X, y, folder_output, file, param_grid, columns)
    
    """
    df = train(X, y, folder_output, file, param_grid, columns)

    if firts :
        df.to_csv(folder_output + ML + ".csv", mode="w", index=False, header=True)
        firts = False
    else : 
        df.to_csv(folder_output + ML + ".csv", mode="a", index=False, header=False)
    """
    
def main():
    columns_lda=["param_LDA__solver", "param_LDA__shrinkage", "param_LDA__store_covariance", "param_LDA__tol", "params", "mean_test_score", "std_test_score"]
    #columns_lda=["param_LDA__solver", "params", "mean_test_score", "std_test_score"]
    #columns_lda=["params", "mean_test_score", "std_test_score"]
    columns_csp=["param_CSP__log", "param_CSP__n_components", "param_CSP__norm_trace", "param_CSP__reg", "params", "mean_test_score", "std_test_score"]
    
    columns =["mean_test_score", "std_test_score"]
    
    logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG, format='%(asctime)s %(message)s')
    
    dataset = "/DATA4/"
    folder_input = "Epoch" + dataset
    folder_output= "Output2" + dataset
    
    """
    logging.info("LDA")
    param_grid = getParamLDA()
    columns = columns_lda
    ML = "LDA"
    """
    
    logging.info("CSP")
    param_grid = getParamCSP()
    columns = columns
    ML = "CSP"
    
    run(folder_input, folder_output, param_grid, columns, ML)

if __name__ == "__main__":
    main()