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
print(round(sample_sizes*0.1))
indice=round(sample_sizes*0.1)-1
print('---------------------')
#A=np.array_split(X, round(sample_sizes*0.1))
X_train=X[:indice,:]
print(X_train.shape)
X_test=X[indice:,:]
print(X_test.shape)
print('---------------------')
y_train=y[:indice]
print(y_train.shape)
y_test=y[indice:]
print(y_test.shape)
print('---------------------')

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size=0.9, random_state=0)
print(X_train_random.shape)
print(X_test_random.shape)
print('---------------------')
print(y_train_random.shape)
print(y_test_random.shape)

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
