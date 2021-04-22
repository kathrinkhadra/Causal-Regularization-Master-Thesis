import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import copy

class causality(object):
    """docstring for causality."""

    def __init__(self, neural_network,new_inputs):
        super(causality, self).__init__()
        self.neural_network = neural_network
        self.new_inputs=new_inputs


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
                self.new_inputs=self.neural_network.net[0:counter](x_train_t)
                covariance=np.cov(self.new_inputs.detach().numpy(),rowvar=False)
                mean=np.array(np.mean(self.new_inputs.detach().numpy(), axis=0))#double check axis maybe more a axis of 1
                #print("mean shape")
                #print(mean.shape)
                #shapes are wrong############################
                #print(self.new_inputs.detach().numpy().shape)

                print("nn generating input")
                print(self.neural_network.net[0:counter])
                print("sliced NN")
                print(new_nn)
                input_samples_ACE=self.ACE(covariance,mean,new_nn)


                #self.plotting_ACE(input_samples_ACE)
                #print("self.new_inputs shape")
                #print(self.new_inputs.detach().numpy().shape)
                #print(mean)
                #print(covariance)

    def ACE(self,covariance,mean,neural_net):
        torch.set_default_dtype(torch.float64)
        input_samples_ACE=[]

        for input_sample in self.new_inputs:
            input_sample=input_sample.detach().numpy()
            #print(input_sample)
            average_causal_effects=[]
            for indx in range(len(input_sample)):
                expectation_do_x = []
                mean_vector=copy.deepcopy(mean)
                #print(mean_vector)
                #print(input_sample)
                mean_vector[indx] = input_sample[indx]#problem 2D = new input and mean vector is not supposed to be
                input_tensor=torch.from_numpy(mean_vector).clone()
                input_tensor.requires_grad=True

                output=neural_net(input_tensor) # torch.from_numpy(mean_vector)

                #output = torch.nn.functional.sigmoid(output)
                #output = torch.nn.functional.softmax(output)

                val = output.data.view(1).cpu().numpy()[0]
                #print("first Loop") #first Loop 0
                #print(indx)
                first_grads = torch.autograd.grad(output, input_tensor, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)
                first_grad_shape = first_grads[0].data.size()
                lower_order_grads = first_grads
                for dim in range(len(mean)):
                    if dim==indx:
                        continue #only executing loop if not same neuron
                    grad_mask = torch.zeros(first_grad_shape)
                    grad_mask[dim] = 1.0
                    #print("lower_order_grads")
                    #print(lower_order_grads)
                    #print("input_tensor")
                    #print(input_tensor)
                    #print("second Loop") #second Loop 1
                    #print(dim)
                    higher_order_grads = torch.autograd.grad(lower_order_grads,input_tensor, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)#allow_unused=False
                    higher_order_grads_array = np.array(higher_order_grads[0].data)

                    temp_cov = copy.deepcopy(covariance)
                    temp_cov[dim][indx] = 0.0
                    val += 0.5*np.sum(higher_order_grads_array*temp_cov[dim])


                average_causal_effects.append(val)


            #average_causal_effects = np.array(average_causal_effects) - np.mean(np.array(average_causal_effects))

            input_samples_ACE.append(np.array(average_causal_effects) - np.mean(np.array(average_causal_effects)))
        #print(average_causal_effects)

        return input_samples_ACE

    def plotting_ACE(self,input_samples_ACE):
        input_samples=self.new_inputs.detach().numpy()
        input_samples=input_samples.T
        for i,causal_effects in enumerate(np.array(input_samples_ACE).T):
            #print(i)
            if i>14:
                break
            #print(causal_effects.shape)
            #print()
            plt.scatter(input_samples[i,:],causal_effects)
            plt.xlabel('Interventional Value')
            plt.ylabel('ACE')
            plt.title('average_causal_effects_neuron_'+str(i))
            plt.savefig('ACEplots/neuron'+str(i)+'.png')
            plt.close()
            plt.hist(causal_effects)
            plt.xlabel('ACE')
            plt.ylabel('Häufigkeit')
            plt.title('histogram_average_causal_effects_neuron_'+str(i))
            plt.savefig('ACEplots/histogram_neuron'+str(i)+'.png')
            plt.close()
        #print(len(input_samples_ACE))
        #print(len(input_samples_ACE[0]))
        print("DONE")
        #print(len(inputs))
        #print(len(inputs[0]))
        #check with sizes what we actually calculated
