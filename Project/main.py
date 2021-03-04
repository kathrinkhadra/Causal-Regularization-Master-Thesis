from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy  as np
import sklearn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pandas as pd

import NeuralNet
import datapreprocessing

######################preprocessing data
test_size=0.9
get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)

###no dataset_shift
get_data.splitting_data_noshift()

#print(get_data.inputs_test.shape)
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

####with datasetshift
#get_data.dataset_shift()
#print(get_data.inputs_test.shape)
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

#####################Neural Net
learning_rate=.0005
epochs=450
neural=NeuralNet.neural_network(learning_rate,0,0,0,epochs,get_data.inputs_training,get_data.target_training,get_data.inputs_test,get_data.target_test)

#build neural net, define optimizer and loss
neural.model(get_data.inputs)

#train neural net
losssave,test_losssave=neural.training()

#plot training performance
#neural.testing_training(losssave,test_losssave,'training_performance.png')

#test neural net
loss_test=neural.testing()

#print(neural.net[0].weight.size())
#print(neural.net[0].bias.size())
print(neural.net)
print(len(neural.net))
