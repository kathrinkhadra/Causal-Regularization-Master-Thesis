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

datasets=[]
get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)

###no dataset_shift
#datasets.append(get_data.splitting_data_noshift())

#print(get_data.inputs_test.shape)
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

####with datasetshift
splits=[394,433,369]
#for split in splits:
#    datasets.append(get_data.dataset_shift(split))#41,243,394,433,369
#print(get_data.inputs_test.shape)
#print(get_data.inputs_training.shape)
#print(get_data.target_training.shape)
#print(get_data.target_test.shape)

#####################Neural Net
learning_rate=.0005
epochs=450
causality_on=0
#factor_list=[1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10]#[1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001]#[10,100,1000,10000]##[0.00001,0.0001,0.001,0.1,1,10,100,1000]#[1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001]#[1e-05]#[0.000000001,0.0000001,0.00001,0.001,0.00000001,0.000001]#np.linspace(0,0.4,10)#[0.1,1,10,100,1000]
#print(factor_list)
factors=[0.0000001,1e-05,1e-08]
factors_ace=[0.00001,0.0001,0.001,0.01,0.1,1,10,100]
if causality_on==0:
    for indx,factor in enumerate(factors):

        get_data.dataset_shift(splits[indx])
        #get_data.target_shift(50)
        #get_data.splitting_data_noshift()
        #get_data.target_shift_big(50) factor=100
        #get_data.target_shift_mid(50,100) factor= 1e-12

        print("CAUSAL NN START")

        txt_file=str(factor)+"_results_datasetshift_regularization_withACE.txt"#str(splits[indx])

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

        #loss_test_causality=neural.testing()

        #PATH=str(splits[indx])+str(factor_ace)+"_model.pth"
        #torch.save(neural.net.state_dict(),PATH)

        f = open(txt_file, 'a')
        f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
        f.close

        print("factor")
        print(factor)
        print("Done")

        del neural.net
        del neural
        torch.cuda.empty_cache()

    get_data.target_shift(50)
    factor=1e-15
    print("CAUSAL NN START")

    txt_file=str(50)+"_results_datasetshift_regularization_withACE.txt"#splits[indx]

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

    #loss_test_causality=neural.testing()

    #PATH=str(splits[indx])+str(factor_ace)+"_model.pth"
    #torch.save(neural.net.state_dict(),PATH)

    f = open(txt_file, 'a')
    f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
    f.close

    print("factor")
    print(factor)
    print("Done")

    del neural.net
    del neural
    torch.cuda.empty_cache()

    get_data.target_shift_big(50)
    factor=100
    print("CAUSAL NN START")

    txt_file=str(50)+"_1_results_datasetshift_regularization_withACE.txt"#splits[indx]

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

    #loss_test_causality=neural.testing()

    #PATH=str(splits[indx])+str(factor_ace)+"_model.pth"
    #torch.save(neural.net.state_dict(),PATH)

    f = open(txt_file, 'a')
    f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
    f.close

    print("factor")
    print(factor)
    print("Done")

    del neural.net
    del neural
    torch.cuda.empty_cache()

    get_data.target_shift_mid(50,100)
    factor= 1e-12
    print("CAUSAL NN START")

    txt_file=str(50)+"_100_1_results_datasetshift_regularization_withACE.txt"#splits[indx]

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

    #loss_test_causality=neural.testing()

    #PATH=str(splits[indx])+str(factor_ace)+"_model.pth"
    #torch.save(neural.net.state_dict(),PATH)

    f = open(txt_file, 'a')
    f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
    f.close

    print("factor")
    print(factor)
    print("Done")

    del neural.net
    del neural
    torch.cuda.empty_cache()

    get_data.splitting_data_noshift()
    factor=1e-15
    print("CAUSAL NN START")

    txt_file=str(0)+"_results_datasetshift_regularization_withACE.txt"#splits[indx]

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

    #loss_test_causality=neural.testing()

    #PATH=str(splits[indx])+str(factor_ace)+"_model.pth"
    #torch.save(neural.net.state_dict(),PATH)

    f = open(txt_file, 'a')
    f.write('loss_test_causal='+str(loss_test_causality)+'\n\n')
    f.close

    print("factor")
    print(factor)
    print("Done")

    del neural.net
    del neural
    torch.cuda.empty_cache()

if False:
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
