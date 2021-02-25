# coding: utf-8
# based on https://discuss.pytorch.org/t/pytorch-fails-to-over-fit-boston-housing-dataset/40365

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy  as np
import sklearn
import matplotlib.pyplot as plt

#import torch
#import torch.nn as nn
import pandas as pd
import seaborn as sns

import random

boston = load_boston()

boston_df = pd.DataFrame(boston['data'] )
boston_df.columns = boston['feature_names']
boston_df['PRICE']= boston['target']

X,y   = (boston.data, boston.target)
dim = X.shape[1] #13

#print(np.amax(y))
#print(np.amin(y))

sample_sizes=X.shape[0]
print(round(sample_sizes*0.1))

sample_sizes=y.shape[0]
#print(round(sample_sizes*0.1))
indice=round(sample_sizes*0.1)-1
#print('---------------------')
#A=np.array_split(X, round(sample_sizes*0.1))
X_train=X[:indice,:]
#print(X_train.shape)
X_test=X[indice:,:]
#print(X_test.shape)
#print('---------------------')
y_train=y[:indice]
#print(y_train.shape)
y_test=y[indice:]
#print(y_test.shape)
#print('---------------------')

finalindices=sample_sizes-indice
X_train=X[finalindices:sample_sizes,:]
#print(X_train.shape)
X_test=X[:finalindices,:]
#print(X_test.shape)
#print('---------------------')
y_train=y[finalindices:sample_sizes]
#print(y_train.shape)
y_test=y[:finalindices]
#print(y_test.shape)
#print('---------------------')

rand=390#369#386#random.randint(indice, finalindices)
#print(rand)

X_train=X[rand:indice+rand,:]
#print(X_train.shape)
X_test=np.concatenate((X[:rand,:],X[indice+rand:,:]))
#print(X_test.shape)
#print('---------------------')
y_train=y[rand:indice+rand]
#print(y_train.shape)
y_test=np.concatenate((y[:rand],y[indice+rand:]))
#print(y_test.shape)
#print('---------------------')


testdata=X_test.T
#print(testdata.shape)
for i, column in enumerate(X_train.T):
    #print(i)
    #print(column)
    #plt.subplot(3,3,i)
#    fig, axes = plt.subplots(1, 2)

    # Train
#    sns.distplot(column, ax=axes[0])
#    axes[0].set_title('Training Data')
    # Test
#    sns.distplot(testdata[i,:], ax=axes[1])
#    axes[1].set_title('Test Data')

#    plt.savefig('figures/datasetshift/distribution'+str(i)+'.png')

    # Train
    sns.displot([column,testdata[i,:]])#sns.distplot(column, hist=False,rug=True)#sns.displot(column, kind="kde")#
    #sns.displot(testdata[i,:])#sns.distplot(testdata[i,:], hist=False,rug=True)#sns.displot(testdata[i,:], kind="kde")#
    plt.savefig('figures/datasetshift/shiftwith390/distribution'+str(i)+'.png')
    plt.close()
#    axes[0].set_title('Training Data')
    # Test
#    sns.distplot(testdata[i,:], ax=axes[1])
#    axes[1].set_title('Test Data')

#    plt.savefig('figures/datasetshift/distribution'+str(i)+'.png')


#X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size=0.9, random_state=0)

#testdata_random=X_test_random.T
#for i, column in enumerate(X_train_random.T):
    #print(i)
    #print(column)
    #plt.subplot(3,3,i)
#    fig, axes = plt.subplots(1, 2)

    # Train
#    sns.distplot(column, ax=axes[0])
#    axes[0].set_title('Training Data')
    # Test
#    sns.distplot(testdata_random[i,:], ax=axes[1])
#    axes[1].set_title('Test Data')

#    plt.savefig('figures/random/distribution'+str(i)+'.png')
#    sns.distplot(column, hist=False, rug=True)
#    sns.distplot(testdata_random[i,:], hist=False, rug=True)
#    plt.savefig('figures/random/distribution'+str(i)+'.png')
#    plt.close()
#print(X_train_random.shape)
#print(X_test_random.shape)
#print('---------------------')
#print(y_train_random.shape)
#print(y_test_random.shape)

#sns_plot=sns.pairplot(boston_df)
#sns_plot.savefig("figures/summary.png")
#for i, column in enumerate(X.T, 1):
    #print(i)
    #print(column)
    #plt.subplot(3,3,i)
#    sns_plot=sns.displot(column)
#    sns_plot.savefig("figures/output"+str(i)+".png")

#sns_plot=sns.displot(X)
#sns_plot.savefig("output.png")
