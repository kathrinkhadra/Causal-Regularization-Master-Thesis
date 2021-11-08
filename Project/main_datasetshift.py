from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from scipy.spatial.distance import jensenshannon
import scipy as sp
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
#get_data.splitting_data_noshift()

#get_data.feature_selection(12)



#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

####with datasetshift
#get_data.dataset_shift(369)#41,243,394,433,369
#from 0 to 12
#433-> Covariate shift in feature (1,2,3,8,9,10) here mutual information was 0
#394 -> Covariate shift in feature (4,5,4,12)
#369 -> Covariate shift in feature (1,2)


#get_data.target_shift(50)
#get_data.target_shift_big(50)
get_data.target_shift_mid(50,100)


#print(get_data.inputs_training.shape)
#print(get_data.inputs_test.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

#print(np.mean(get_data.inputs_test)-np.mean(get_data.inputs_training))
#print(np.mean(get_data.target_test)-np.mean(get_data.target_training))
#a, inputs_test, b, target_test = train_test_split(get_data.inputs_test, get_data.target_test , test_size=50, random_state=0)
#print(target_test)
#print(get_data.target_training)
#print(normalized_mutual_info_score(target_test,get_data.target_training,average_method='arithmetic'))
#print(mutual_info_score(inputs_test[:,8],get_data.inputs_training[:,8]))
#for i in range(13):
    #print(inputs_test[:,i].shape)
    #print(get_data.inputs_training[:,i].shape)
#    print(i)
#    print(normalized_mutual_info_score(inputs_test[:,i],get_data.inputs_training[:,i]))
    #print(inputs_test[:,i])
    #print(get_data.inputs_training[:,i])
    #jsd(inputs_test[:,i],get_data.inputs_training[:,i])
    #jensenshannon(inputs_test[:,i],get_data.inputs_training[:,i], axis=1)
#mutual_score=[]
#for i in range(450):
#    get_data.target_shift(i)
#    a, inputs_test, b, target_test = train_test_split(get_data.inputs_test, get_data.target_test , test_size=50, random_state=0)
#    mutual_score.append(normalized_mutual_info_score(target_test,get_data.target_training,average_method='arithmetic'))


#print(np.argmin(mutual_score))
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)


#####################Neural Net
learning_rate=.0005
epochs=450
causality_on=1
factors=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]

if causality_on==1:
    for factor in factors:

        print("CAUSAL NN START")

        txt_file=str(factor)+"_results_datasetshift_regularization_withACE.txt"

        f = open(txt_file, 'a')
        f.write('-------------------------------CAUSAL NEURAL NET-------------------------------\n\n')
        f.close

        neural=NeuralNet.neural_network(learning_rate,0,0,0,epochs,get_data.inputs_training,get_data.target_training,get_data.inputs_test,get_data.target_test,causality_on,txt_file,0,factor,0,[])

        #build neural net, define optimizer and loss
        neural.model(get_data.inputs)

        #train neural net
        loss_training,test_loss_training=neural.training()

        #plot training performance
        #neural.testing_training(loss_training,test_loss_training,'training_performance.png')

        #test neural net
        loss_test_causality=neural.testing()

        PATH=str(factor)+"_model.pth"

        #torch.save(neural.net.state_dict()﻿,PATH)
        torch.save(neural.net.state_dict(),PATH)

        f = open(txt_file, 'a')
        f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
        f.close

        print("factor")
        print(factor)
        print("Done")

        del neural.net
        del neural
        torch.cuda.empty_cache()


if True:
    print("NORMAL NN START")

    txt_file="results_datasetshift_regularization_noACE.txt"

    f = open(txt_file, 'a')
    f.write('------------------------------CONTROL NEURAL NET-------------------------------\n\n')
    f.close

    learning_rate=.0005
    epochs=450
    neural_controll=NeuralNet.neural_network(learning_rate,0,0,0,epochs,get_data.inputs_training,get_data.target_training,get_data.inputs_test,get_data.target_test,0,txt_file,0,10,0,[])

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
