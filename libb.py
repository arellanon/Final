#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:37:02 2021

@author: nahuel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:39:10 2021

@author: nahuel
"""
import os
import configparser
from os import listdir
from os.path import isfile, isdir

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)
from matplotlib import mlab
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg
import numpy as np
import scipy.io



def ls1(path):
    lista = []
    if os.path.exists(path) and os.path.isdir(path):
        lista = [obj for obj in listdir(path) if isdir(path + obj)]
    return lista

def filtro(lista, inicial):
    result=[]
    for a in lista:
        if a[:len(inicial)]==inicial:
            result.append( int(a[1:]) )
    return result

def new_name(path, inicial):
    #Calculamos directorio enumerado    
    directorios = ls1(path)
    #filtramos los directorios que empiezan con T
    lista = filtro(directorios, inicial)
    num=1
    if lista:
        num = max(lista) + 1
    name = inicial + str(num)
    return name

def loadDatos(data_cnt, ch_name_file):
    #Seteamos frecuencia de muestreo Cyton
    freq=250
    #Se carga la matriz de datos
    #data_cnt=data_cnt.transpose()
    #print("data_cnt: ", data_cnt.shape)
    
    #Se carga los nombre de los caneles
    ch_names_txt = open(ch_name_file, "r")
    ch_names = ch_names_txt.read().split(',')
    for i in range(len(ch_names)):
        ch_names[i]=ch_names[i].strip()
    info = mne.create_info(ch_names, freq, 'eeg')
    raw = mne.io.RawArray(data_cnt, info, first_samp=0, copy='auto', verbose='critical')
    #print(info)    
    return raw

def saveConfig(filename, section, configData):
    config = configparser.ConfigParser()
    config.read(filename)
    config[section] = configData
    #print(type(a))
    with open(filename, 'w') as configfile:
        config.write(configfile)

def loadConfig(filename, section):
    config = configparser.ConfigParser()
    config.read(filename)
    calibration = config[section]
    configData = {}
    for key, value in calibration.items():            
        if key == 'path':
            configData[key] = value
        elif key == 'tmin':
            configData[key] = float(value)
        elif key == 'tmax':
            configData[key] = float(value)
        else:
            configData[key] = int(value)
    return configData

def psd(trials, sample_rate):
    '''
    Calculates for each trial the Power Spectral Density (PSD).
    
    Parameters
    ----------
    trials : 3d-array (trials x channels x samples)
        The EEG signal
    
    Returns
    -------
    trial_PSD : 3d-array (trials x channels x PSD)
        the PSD for each trial.  
    freqs : list of floats
        The frequencies for which the PSD was computed (useful for plotting later)
    '''
    
    ntrials = trials.shape[0]
    nchannels = trials.shape[1]
    nsamples = trials.shape[2]
    
    nsample_psd = (nsamples // 2) + 1
    
    trials_PSD = np.zeros((ntrials, nchannels, nsample_psd))

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[trial, ch,:], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[trial, ch, :] = PSD.ravel()
                
    return trials_PSD, freqs

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().
    
    Parameters
    ----------
    trials : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd() 
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))
    
    nchans = len(chan_ind)
    
    # Maximum of 3 plots per row
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)
    
    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows,ncols,i+1)
    
        # Plot the PSD for each class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][:,ch,:], axis=0), label=cl) 
    
        # All plot decoration below...
        
        plt.xlim(1,30)
        
        if maxy != None:
            plt.ylim(0,maxy)
    
        plt.grid()
    
        plt.xlabel('Frequency (Hz)')
        
        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()
        
    plt.tight_layout()
    
def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[0]
    nchannels = trials.shape[1]
    nsamples = trials.shape[2]
    trials_filt = np.zeros((ntrials, nchannels, nsamples))
    for i in range(ntrials):
        trials_filt[i,:,:] = scipy.signal.filtfilt(a, b, trials[i,:,:], axis=1)
    
    return trials_filt

# Calculate the log(var) of the trials
def logvar(trials):
    '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
    return np.log(np.var(trials, axis=2))

def plot_logvar(trials, nchannels, cl_lab):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12,5))
    
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl_lab[0]], axis=0)
    y1 = np.mean(trials[cl_lab[1]], axis=0)

    plt.bar(x0, y0, width=0.5, color='r')
    plt.bar(x1, y1, width=0.4, color='b')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.title("Barras")
    plt.legend(cl_lab)

def cov(trials):
    ''' Calculate the covariance for each trial and return their average '''
    ntrials = trials.shape[0]
    nsamples = trials.shape[2]
    print("ntrials: ",ntrials)
    print("nsamples: ",nsamples)
    test =  trials[0,:,:].dot(trials[0,:,:].T)
    print("test ",test.shape)
    covs = [ trials[i,:,:].dot(trials[i,:,:].T) / nsamples for i in range(ntrials) ]
    print("covs: ", (covs[0][0]) )
    return np.mean(covs, axis=0)

def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = linalg.svd(sigma)
    return U.dot( np.diag(l ** -0.5) )

def csp(trials_r, trials_f):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_l - Array (channels x samples x trials) containing left hand movement trials
        trials_r - Array (channels x samples x trials) containing rigth hand movement trials
    returns:
        Mixing matrix W
    '''
    print("trials_r: ", trials_r.shape)
    print("trials_f: " ,trials_f.shape)
    cov_r = cov(trials_r)
    cov_f = cov(trials_f)
    
    print("cov_r: ", cov_r.shape)
    print("cov_f: ", cov_f.shape)

    #print("cov_r: ", cov_r)
    #print("cov_f: ", cov_f)    
    
    #print("sigma: ", cov_r + cov_f)
    P = whitening(cov_r + cov_f)
    print("P: ", P.shape)
    
    B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
    print("B: ", B.shape)
    
    W = P.dot(B)
    print("W: ", W.shape)
    
    return W

def apply_mix(W, trials):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    ntrials = trials.shape[0]
    nchannels = trials.shape[1]
    nsamples = trials.shape[2]
    trials_csp = np.zeros((ntrials, nchannels, nsamples))
    for i in range(ntrials):
        trials_csp[i,:,:] = W.T.dot(trials[i,:,:])
    return trials_csp

def plot_scatter(left, foot, cl_lab):
    plt.figure()
    plt.scatter(left[:,0], left[:,-1], color='r')
    plt.scatter(foot[:,0], foot[:,-1], color='b')
    plt.xlabel('Last component')
    plt.ylabel('First component')
    plt.legend(cl_lab)
    
def train_lda(class1, class2):
    '''
    Trains the LDA algorithm.
    arguments:
        class1 - An array (observations x features) for class 1
        class2 - An array (observations x features) for class 2
    returns:
        The projection matrix W
        The offset b
    '''
    nclasses = 2
    
    nclass1 = class1.shape[0]
    nclass2 = class2.shape[0]
    
    # Class priors: in this case, we have an equal number of training
    # examples for each class, so both priors are 0.5
    prior1 = nclass1 / float(nclass1 + nclass2)
    prior2 = nclass2 / float(nclass1 + nclass1)
   
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    
    class1_centered = class1 - mean1
    class2_centered = class2 - mean2
    
    # Calculate the covariance between the features
    cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
    cov2 = class2_centered.T.dot(class2_centered) / (nclass2 - nclasses)
   
    W = (mean2 - mean1).dot(np.linalg.pinv(prior1*cov1 + prior2*cov2))
    b = (prior1*mean1 + prior2*mean2).dot(W)
    
    return (W,b)

def apply_lda(test, W, b):
    '''
    Applies a previously trained LDA to new data.
    arguments:
        test - An array (features x trials) containing the data
        W    - The project matrix W as calculated by train_lda()
        b    - The offsets b as calculated by train_lda()
    returns:
        A list containing a classlabel for each trial
    '''
    ntrials = test.shape[1]
    
    prediction = []
    for i in range(ntrials):
        # The line below is a generalization for:
        # result = W[0] * test[0,i] + W[1] * test[1,i] - b
        result = W.dot(test[:,i]) - b
        if result <= 0:
            prediction.append(1)
        else:
            prediction.append(2)
    
    return np.array(prediction)