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

print(np.abs(np.mean(get_data.inputs_test)-np.mean(get_data.inputs_training)))
print(np.abs(np.mean(get_data.target_test)-np.mean(get_data.target_training)))


print(np.abs(np.var(get_data.inputs_test)-np.var(get_data.inputs_training)))
print(np.abs(np.var(get_data.target_test)-np.var(get_data.target_training)))

#print(get_data.inputs_test.shape)
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

#0.010215797229068846
#0.031188693957115043
#0.0016818611924868526
#0.012932410147243781
#shift
#0.17324271391979196
#0.1847641325536063
#0.031057891081192293
#0.028048193168648296
print("shift")
####with datasetshift
x=[0:490]
input_mean_array=[]
target_mean_array=[]
input_var_array=[]
target_var_array=[]
for i in x:
    get_data.dataset_shift(i)#400,432   #169,220,210,250
    #print(get_data.inputs_test.shape)
    #print(get_data.inputs_training.shape)
    #print(get_data.target_training.shape)
    #print(get_data.target_test.shape)

    input_mean=np.abs(np.mean(get_data.inputs_test)-np.mean(get_data.inputs_training))
    target_mean=np.abs(np.mean(get_data.target_test)-np.mean(get_data.target_training))


    input_var=np.abs(np.var(get_data.inputs_test)-np.var(get_data.inputs_training))
    target_var=np.abs(np.var(get_data.target_test)-np.var(get_data.target_training))

    input_mean_array.append(input_mean)
    target_mean_array.append(target_mean)
    input_var_array.append(input_var)
    target_var_array.append(input_var)

i=np.argmax(input_mean_array)

print(i)

#####################Neural Net
learning_rate=.0005
epochs=450
causality_on=0

print("CAUSAL NN START")

if causality_on==1:

    txt_file="results_NOdatasetshift_regularization_withACE.txt"

    f = open(txt_file, 'a')
    f.write('-------------------------------CAUSAL NEURAL NET-------------------------------\n\n')
    f.close

    neural=NeuralNet.neural_network(learning_rate,0,0,0,epochs,get_data.inputs_training,get_data.target_training,get_data.inputs_test,get_data.target_test,causality_on,txt_file,0,10000,0)

    #build neural net, define optimizer and loss
    neural.model(get_data.inputs)

    #train neural net
    loss_training,test_loss_training=neural.training()

    #plot training performance
    #neural.testing_training(loss_training,test_loss_training,'training_performance.png')

    #test neural net
    loss_test_causality=neural.testing()
    f = open(txt_file, 'a')
    f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
    f.close

    print("NORMAL NN START")

    txt_file="results_NOdatasetshift_regularization_noACE.txt"

    f = open(txt_file, 'a')
    f.write('------------------------------CONTROL NEURAL NET-------------------------------\n\n')
    f.close

    learning_rate=.0005
    epochs=450
    neural_controll=NeuralNet.neural_network(learning_rate,0,0,0,epochs,get_data.inputs_training,get_data.target_training,get_data.inputs_test,get_data.target_test,0,txt_file,0,10000,0)

    #build neural net, define optimizer and loss
    neural_controll.model(get_data.inputs)

    #train neural net
    loss_training,test_loss_training=neural_controll.training()

#plot training performance
#neural_controll.testing_training(loss_training,test_loss_training,'training_performance_control.png')

#test neural net
    loss_test=neural_controll.testing()
    f = open(txt_file, 'a')
    f.write('loss_test_control='+str(loss_test)+'\n\n')
    f.close

#causal_test=causal.causality(neural_controll,0,0,0,0,0,0,txt_file)

#causal_test.slicing_NN(get_data.inputs_training)#for us only training values as we are covering ACE in training not after training

#print("final causality")
#print(causal_test.final_causality)


#print("shapes")
#for row in causal_test.final_causality:
#    element=np.array(row)
#    print(len(row))
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
