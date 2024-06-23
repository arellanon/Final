#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

@author: nahuel
"""
from libb import *
import numpy as np
import scipy.io
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations

def main():
    trials1, info1 = get_epoch("Epoch/DATA4/", "D4SBI_S02")
    #print(trials1['left'][0][1])
    show_info(trials1, info1)
    report_psd(trials1, info1)
    calculateML(trials1, info1, True)
    
    trials2, info2 = get_epoch("Epoch/DATA3/", "D3S02")
    #print(trials2['left'][0][1])
    show_info(trials2, info2)
    report_psd(trials2, info2)
    calculateML(trials2, info2, True)
    
## Funciones reporte    
def show_info(trials, info):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    print('Sample rate:', sample_rate)
    print('Number of channels:', nchannels)
    print('Channel names:', channel_names)
    print('Class labels:', cl_lab)
    print('Shape ', cl1,' :', trials[cl1].shape)
    print('Shape ', cl2,' :', trials[cl2].shape)
    
def report_psd(trials, info):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    print("channels: ", channel_names)
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    # PSD epoch
    psd_l, freqs = psd(trials[cl1], sample_rate)
    psd_r, freqs = psd(trials[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right']
        #maxy=500
    )
    
    # PSD epoch bandpass
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                   cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    
    psd_l, freqs = psd(trials_filt[cl1], sample_rate)
    psd_r, freqs = psd(trials_filt[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right']
        #maxy=300
    )
    
    # PSD epoch csp
    W = csp(trials_filt[cl1], trials_filt[cl2])
    
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                  cl2: apply_mix(W, trials_filt[cl2])}
    
    psd_l, freqs = psd(trials_csp[cl1], sample_rate)
    psd_r, freqs = psd(trials_csp[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}

    plot_psd(
        trials_PSD, 
        freqs, 
        [0, nchannels // 2 ,-1],
        chan_lab=['first component', 'middle component', 'last component']
        #maxy=0.75 
    )
    
    #Logvar
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                     cl2: logvar(trials_filt[cl2])}
    
    plot_logvar(trials_logvar, nchannels, cl_lab)

    trials_csp_logvar = {cl1: logvar(trials_csp[cl1]),
                     cl2: logvar(trials_csp[cl2])}
    
    plot_logvar(trials_csp_logvar, nchannels, cl_lab)
    
    # Scatterplot
    plot_scatter(trials_csp_logvar[cl1], trials_csp_logvar[cl2], cl_lab)    
    
def calculateML(trials, info, flag_plot):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    # PSD epoch bandpass
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                   cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    
    train, test = prepareData(trials_filt, cl_lab)
    
    #print("train: ", train['left']) 
    #print("test: ", test['left'].shape) 
    
    W,b = train_lda(train[cl1], train[cl2])
    print('W:', W)
    print('b:', b)
        
    # Scatterplot
    #plot_scatter(trials_logvar[cl1], trials_logvar[cl2], cl_lab)
    
    # Scatterplot like before
    if flag_plot :
        plot_scatter(train[cl1], train[cl2], cl_lab)
        
        # Calculate decision boundary (x,y)
        x = np.arange(-5, 1, 0.1)
        y = (b - W[0]*x) / W[1]
        
        # Plot the decision boundary
        plt.plot(x, y, linestyle='--', linewidth=2, color='k')
        #plt.xlim(-5, 1)
        #plt.ylim(-2.2, 1)
        
        plot_scatter(test[cl1], test[cl2], cl_lab)
        #title('Test data')
        plt.plot(x,y, linestyle='--', linewidth=2, color='k')
        #plt.xlim(-5, 1)
        #plt.ylim(-2.2, 1)
    

    # Print confusion matrix
    conf = np.array([
        [(apply_lda(test[cl1].T, W, b) == 1).sum(), (apply_lda(test[cl2].T, W, b) == 1).sum()],
        [(apply_lda(test[cl1].T, W, b) == 2).sum(), (apply_lda(test[cl2].T, W, b) == 2).sum()],
    ])
    
    print('Confusion matrix:')
    print(conf)
    print()
    print('Accuracy: %.3f' % (np.sum(np.diag(conf)) / float(np.sum(conf))))
    print()
    
def prepareData(trials_filt, cl_lab):
    train_percentage = 0.8
    cl1=cl_lab[0]
    cl2=cl_lab[1]
    
    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_l = int(trials_filt[cl1].shape[0] * train_percentage)
    ntrain_r = int(trials_filt[cl2].shape[0] * train_percentage)
    ntest_l = trials_filt[cl1].shape[0] - ntrain_l
    ntest_r = trials_filt[cl2].shape[0] - ntrain_r
    
    
    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:ntrain_l,:,:],
             cl2: trials_filt[cl2][:ntrain_r,:,:]}
    
    test = {cl1: trials_filt[cl1][ntrain_l:,:,:],
            cl2: trials_filt[cl2][ntrain_r:,:,:]}
    
    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])
    
    print("W:", W.shape)
    
    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])
    
    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][:,comp,:]
    train[cl2] = train[cl2][:,comp,:]
    test[cl1] = test[cl1][:,comp,:]
    test[cl2] = test[cl2][:,comp,:]
    
    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])
    
    return train, test
    

### Funciones get_data
def get_epoch(path, filename):
    epochs=mne.read_epochs(path + filename + "-epo.fif", proj=True, preload=True, verbose=None)
    trials = {}
    info = {}
    channel_names = epochs.info['ch_names']
    sample_rate = epochs.info['sfreq']

    cl_lab=list(epochs.event_id.keys())
    keys=list(epochs.event_id.values())
        
    epochs0=epochs[epochs.events[:,2]==keys[0]]
    epochs1=epochs[epochs.events[:,2]==keys[1]]
    
    trials[cl_lab[0]] = epochs0.get_data(units='uV')
    trials[cl_lab[1]] = epochs1.get_data(units='uV')
    
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    info['channel_names'] = channel_names
    
    return trials, info

if __name__ == "__main__":
    main()