# -*- coding: utf-8 -*-
from numpy import *
from mvpa.suite import *
import pylab as P
import os
from scipy import signal

verbose.level = 4

## prepare data
def loadData(name):
    # load prepared data (eyeblink removal)
    data = loadtxt('subj/'+name+'.txt')
    labels = loadtxt('subj/'+name+'_attrib.txt')
    # samplerate (350Hz) * 3sec
    n = 1050
    # 144 samples with 2 features
    presamples = zeros((144,n,2))
    for i in range(144):
	presamples[i] = data[data[:,2]==i+1,0:2][0:n]

# downsampling parameters
    resa = 100.
    
# time courses of coordinates and velocity
    l = labels.shape[0]
    tdata = ones((l,resa,5))
    for i in range(l):
	# x-coordinate
        tdata[i,:,0] = signal.resample(presamples[i,:,0],resa,window = 'hann')
        # y-coordinate
	tdata[i,:,1] = signal.resample(presamples[i,:,1],resa,window = 'hann')
        # x-velocity
	tdata[i,1:,2] = diff(tdata[i,:,0])
        # y-velocity
	tdata[i,1:,3] = diff(tdata[i,:,1])
        # total velocity
	tdata[i,:,4] = sqrt(tdata[i,:,2]**2+tdata[i,:,3]**2)
    
    samples = tdata

#create chunks
    chunks = zeros(l)
    for i in range(len(unique(labels))):
        h = array(nonzero(labels ==i))
        chunks[h] = arange(l/len(unique(labels)))%8

# plot samples
    printer = 0
    if printer:
	c = ('m','r','y','g','k')  
	for j in range(5):
	    P.figure()
	    for i in range(2):
		plotERP(samples[labels[:,0]==i+1,:,j],pre_mean = 0, pre_onset = 0, post = 100, SR = 1, errtype = 'ci95', color = c[i], errcolor = c[i])
    return (samples, labels, chunks)

## create dataset and classification
def doit((samples,attrib,chunks),sf):

# select features and labels
    samples = samples[:,:,sf]
    # 1st label: inverted upright
    # 2nd label: male-female
    # 3rd label: identity
    label = attrib[:,0]

# mirror y-coordinates
    h = 1
    if h:
	samples[label==1,:,1] = (samples[label==1,:,1]-512)*-1+512

# create dataset
    dataset = MaskedDataset(samples=samples,
                            labels=label,
                            chunks=chunks)

# plot samples per label
    n = [640,512]
    achsen = ['x','y']
    c = ('r','b','y','g','m')
    for j in range(len(sf)):
        P.figure()
        for i in range(len(unique(dataset.labels))):
	#for i in range(1):
	    data = dataset.O[dataset.labels==i+1,:,j]
            plotERP(data,pre_mean = 0, pre = 0, post = 100, SR = 1, errtype = 'std', color = c[i], errcolor = c[i])
	    P.title(achsen[j]+'-coordinates')
	    P.xlabel('time [ms]')
	    P.ylabel('coordinate [px]')
	    P.xticks(arange(6)*20,arange(6)*20*30)
	    if j==0:
		P.axis([0,100,n[j]-100,n[j]+100])
	    else:
		P.axis([0,100,n[j]+100,n[j]-100])

	
# plot picture	
    O = P.imread('f05.png')

    mx1 = mean(dataset.O[dataset.labels==1,:,0], axis = 0)-640+O.shape[1]/2
    my1 = mean(dataset.O[dataset.labels==1,:,1], axis = 0)-512+O.shape[0]/2
    mx2 = mean(dataset.O[dataset.labels==2,:,0], axis = 0)-640+O.shape[1]/2
    my2 = mean(dataset.O[dataset.labels==2,:,1], axis = 0)-512+O.shape[0]/2

    P.figure()
    P.imshow(O)
    P.plot(mx1,my1,c[0])
    P.plot(mx2,my2,c[1])
    P.xticks(arange(8)*50+150,arange(8)*50+490)
    P.yticks(arange(8)*50,arange(8)*50+312)
    P.axis([150,450,380,20])
    P.show()
     
# zcore per feature
    zscore(dataset)
    removeInvariantFeatures(dataset)
    
# classification
    cf = SVM()
    
    tfe = TransferError(cf)
    cv = CrossValidatedTransferError(tfe,
                                    NFoldSplitter(cvtype=1),
                                    enable_states = ['confusion',
                                                    'training_confusion',
                                                    'transerrors',
                                                    'null_prob',
                                                    'samples_error',
                                                    'results',
                                                    ],
                                    combiner = array,
                                    )
    errors = cv(dataset)
    
# plot results
    print 'Classifier: ', 1-mean(errors)
    print cv.confusion
    
# get sensitivities
    sensana = cf.getSensitivityAnalyzer()
    sens = dataset.mapReverse(sensana(dataset))

#plot standardized sensitivities
    for j in range(len(sf)):
        P.figure()
	P.plot(sens[:,j]/max(sens[:,j]),'k')
	P.axis([0,100,-1,1])
	P.title(achsen[j]+'-coordinates')
	P.xlabel('time')
	P.xticks(arange(6)*20,arange(6)*20*30)
    print  mean(sens,axis = 0), '\n'
    return (cv,sens,dataset)
        

## script
D = loadData('jate1')
W = doit(D,[0,1])
