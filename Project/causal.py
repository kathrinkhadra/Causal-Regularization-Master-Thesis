import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt

class causality(object):
    """docstring for causality."""

    def __init__(self, neural_network):
        super(causality, self).__init__()
        self.neural_network = neural_network


    def slicing_NN(self,input_sample):
        #slicing after the unlinearity needs to be done
        #y_train_t =torch.from_numpy(self.target_training).clone().reshape(-1, 1)
        x_train_t =torch.from_numpy(input_sample).clone()
        for counter,layer in enumerate(self.neural_network.net):
            if counter%2==0 and counter<len(self.neural_network.net): #wrong slicing
                #print(counter)
                new_nn=self.neural_network.net[counter:len(self.neural_network.net)]
                new_input=self.neural_network.net[0:counter](x_train_t) #passing a numpy array -> need to change that
                #covariance=np.cov(new_input) # need to convert to numpy array
                #mean=np.mean(new_input)# need to convert to numpy array
                print(new_nn)
                #print(mean)
                #print(covariance)
