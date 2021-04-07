import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import copy

class causality(object):
    """docstring for causality."""

    def __init__(self, neural_network):
        super(causality, self).__init__()
        self.neural_network = neural_network


    def slicing_NN(self,input_sample):
        #slicing after the unlinearity needs to be done
        #y_train_t =torch.from_numpy(self.target_training).clone().reshape(-1, 1)
        #print(input_sample.shape)
        x_train_t =torch.from_numpy(input_sample).clone()
        for counter,layer in enumerate(self.neural_network.net):
            if counter%2==0 and counter<len(self.neural_network.net):
                #print(counter)
                self.neural_network.net.eval()
                new_nn=self.neural_network.net[counter:len(self.neural_network.net)]
                new_nn.eval()
                new_input=self.neural_network.net[0:counter](x_train_t)
                covariance=np.cov(new_input.detach().numpy(),rowvar=False)
                mean=np.array(np.mean(new_input.detach().numpy(), axis=0))#double check axis maybe more a axis of 1
                #print("mean shape")
                #print(mean.shape)
                #shapes are wrong############################
                #print(new_input.detach().numpy().shape)

                self.ACE(covariance,mean,new_nn,new_input)
                #print("nn generating input")
                #print(self.neural_network.net[0:counter])
                #print("new_input shape")
                #print(new_input.detach().numpy().shape)
                #print(mean)
                #print(covariance)

    def ACE(self,covariance,mean,neural_net,inputs):
        torch.set_default_dtype(torch.float64)
        for input_sample in inputs:
            input_sample=input_sample.detach().numpy()

            for indx in range(len(input_sample)):
                expectation_do_x = []
                mean_vector=copy.deepcopy(mean)
                #print(mean_vector)
                #print(input_sample)
                mean_vector[indx] = input_sample[indx]#problem 2D = new input and mean vector is not supposed to be
                output=neural_net(torch.from_numpy(mean_vector)) # does this make sense?
                input_tensor=torch.from_numpy(mean_vector).clone()
                input_tensor.requires_grad=True

                val = output.data.view(1).cpu().numpy()[0]
                
                first_grads = torch.autograd.grad(output, input_tensor, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)#allow_unused=True
                first_grad_shape = first_grads[0].data.size()
                lower_order_grads = first_grads
                for dim in range(len(mean)):
                    if dim==indx:
                        continue #only executing loop if not same neuron
                    grad_mask = torch.zeros(first_grad_shape)
                    grad_mask[dim] = 1.0

                    higher_order_grads = torch.autograd.grad(lower_order_grads,input_tensor, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)#allow_unused=False
                    higher_order_grads_array = np.array(higher_order_grads[0].data)

                    temp_cov = copy.deepcopy(covariance)
                    temp_cov[dim][indx] = 0.0
                    val += 0.5*np.sum(higher_order_grads_array*temp_cov[dim])


            #average_causal_effects.append(val)


        #average_causal_effects = np.array(average_causal_effects) - np.array(baseline_expectation_do_x)[:len(average_causal_effects)]
