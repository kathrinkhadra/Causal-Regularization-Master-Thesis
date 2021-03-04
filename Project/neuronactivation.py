import torch
import torch.nn as nn
import numpy  as np
import math


class NeuronActivation(object):
    """docstring for NeuronActivation."""

    def __init__(self, net):
        super(NeuronActivation, self).__init__()
        self.net = net

    def extract_activation(self):
        #summary(self.net)
        num_layers=math.ceil(len(neural.net)/2)
        for i in range(0,num_layers-1):
            self.net[i].weight
