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
import causal

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
loss_training,test_loss_training=neural.training()

#plot training performance
#neural.testing_training(loss_training,test_loss_training,'training_performance.png')

#test neural net
loss_test=neural.testing()

causal_test=causal.causality(neural,0)

causal_test.slicing_NN(get_data.inputs_training)#for us only training values as we are covering ACE in training not after training

#print(neural.net[0].weight.size())
#print(neural.net[0].bias.size())
#print(neural.net[0:5])#abscheiden des netzes
#print(len(neural.net))
#counter=1
#new_nn=neural.net[counter:len(neural.net)]
#new_input=neural.net[0:counter]#(get_data.inputs_training)


#print(len(get_data.inputs_training))
#print(len(get_data.inputs_training[0]))
#print(neural.net(get_data.inputs_training))
#print(new_nn)
#print(new_input)
