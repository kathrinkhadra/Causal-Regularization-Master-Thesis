import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import copy

class causality(object):
    """docstring for causality."""

    def __init__(self, neural_network,new_inputs,input_samples_ACE,counter,final_causality):
        super(causality, self).__init__()
        self.neural_network = neural_network
        self.new_inputs=new_inputs
        self.input_samples_ACE=input_samples_ACE
        self.counter=counter
        self.final_causality=final_causality


    def slicing_NN(self,input_sample):
        #slicing after the unlinearity needs to be done
        #y_train_t =torch.from_numpy(self.target_training).clone().reshape(-1, 1)
        #print(input_sample.shape)
        self.final_causality=[]
        x_train_t =torch.from_numpy(input_sample).clone()
        for counter,layer in enumerate(self.neural_network.net):
            if counter%2==1 and counter<len(self.neural_network.net)-1 and counter!=0:#if counter==6:#
                self.counter=counter
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

                self.ACE(covariance,mean,new_nn)

                causality_update=self.evaluating_ACE()

                self.final_causality.append(causality_update)

                #self.final_causality.append(causality_update, dtype=object)
                #if counter==0:
                #self.plotting_ACE()
                #print("self.new_inputs shape")
                #print(self.new_inputs.detach().numpy().shape)
                #print(mean)
                #print(covariance)
        self.final_causality=np.array(self.final_causality,dtype=object)

    def ACE(self,covariance,mean,neural_net):
        torch.set_default_dtype(torch.float64)
        self.input_samples_ACE=[]
        first_orders_mean_array=[]
        high_orders_mean_array=[]
        for input_sample in self.new_inputs:
            first_orders_mean=[]
            high_orders_mean=[]
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
                first_grads = torch.autograd.grad(output, input_tensor, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)#allow_unused=False
                #first_grads=np.array(first_grads, dtype=float)
                #print(first_grads)
                #first_grads[0].data=torch.nan_to_num(first_grads[0].data)
                #print(first_grads)
                #first_grads=torch.tensor(first_grads)
                first_grad_shape = first_grads[0].data.size()
                lower_order_grads = first_grads
                first_orders_mean.append(np.mean(np.array(first_grads[0].data)))
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
                    #higher_order_grads=np.array(higher_order_grads, dtype=float)
                    #print(higher_order_grads)
                    #X=torch.nan_to_num(higher_order_grads[0].data)
                    #print(X)
                    #higher_order_grads=torch.tensor(higher_order_grads)
                    higher_order_grads_array = np.array(higher_order_grads[0].data)
                    high_orders_mean.append(np.mean(higher_order_grads_array))

                    temp_cov = copy.deepcopy(covariance)
                    temp_cov[dim][indx] = 0.0
                    val += 0.5*np.sum(higher_order_grads_array*temp_cov[dim])


                average_causal_effects.append(val)


            #average_causal_effects = np.array(average_causal_effects) - np.mean(np.array(average_causal_effects))
                first_orders_mean_array.append(first_orders_mean)
                high_orders_mean_array.append(high_orders_mean)
            self.input_samples_ACE.append(np.array(average_causal_effects) - np.mean(np.array(average_causal_effects)))
        #print(average_causal_effects)

            #self.plotting_derivatives(first_orders_mean_array, high_orders_mean_array)


    def evaluating_ACE(self):
        ACEs=np.array(self.input_samples_ACE).T
        medians=np.median(ACEs, axis=1)
        variances=np.var(ACEs, axis=1)
        plus_border_mean=np.percentile(medians,80)
        minus_border_mean=np.percentile(medians,20)
        plus_border_variances=np.percentile(variances,80)
        minus_border_variances=np.percentile(variances,20)

        #medians[medians>=plus_border_mean]=0
        #medians[medians<plus_border_mean]=1
        #medians[medians<=minus_border_mean]=0
        #medians[medians>minus_border_mean]=1
        #print("get causality")
        #print(plus_border_mean)
        #print(minus_border_mean)
        #print(medians)

        causality_update = [0 if med>=plus_border_mean or med<=minus_border_mean else 1 for med in medians]
        #print(medians)
        #point=np.median(medians)
        #print(point)
        #print(medians.shape)
        #print(variances.shape)
        #print("median")
        #print(medians[0])
        #first_row=ACEs[0,:]
        #print("firstrow")
        #print(first_row)
        #print("median first row")
        #print(np.median(first_row))
        #for i,causal_effects in enumerate(np.array(self.input_samples_ACE).T):
    #        if i!=0:
    #            break
    #        print("causal effects")
    #        print(causal_effects)
        #self.plotting_ACE_mean_var(medians,variances)
        return causality_update

    def plotting_ACE_mean_var(self,medians,variances):
        plt.hist(medians)
        #point=np.median(medians)
        plt.annotate("mean", xy=(np.mean(medians),0),arrowprops = dict(facecolor='black', shrink=0.05))
        #plt.annotate("border-", xy=(np.max(medians)-0.2*np.max(medians),0),arrowprops = dict(facecolor='black', shrink=0.05))
        #plt.annotate("border+", xy=(np.min(medians)+0.2*np.min(medians),0),arrowprops = dict(facecolor='black', shrink=0.05))
        plt.annotate("border+", xy=(np.percentile(medians,80),0),arrowprops = dict(facecolor='black', shrink=0.05))
        plt.annotate("border-", xy=(np.percentile(medians,20),0),arrowprops = dict(facecolor='black', shrink=0.05))

        #plt.xlabel('Neuron')
        #plt.ylabel('Mean Value')
        plt.title('Mean Values of Input Neurons')
        plt.savefig('MeanVarPlots/mean_neuron_'+str(self.counter)+'.png')
        plt.close()
        plt.hist(variances)
        plt.annotate("median", xy=(np.median(variances),0),arrowprops = dict(facecolor='black', shrink=0.05))
        plt.annotate("border+", xy=(np.percentile(variances,80),0),arrowprops = dict(facecolor='black', shrink=0.05))
        plt.annotate("border-", xy=(np.percentile(variances,20),0),arrowprops = dict(facecolor='black', shrink=0.05))
        #plt.annotate("border-", xy=(np.max(variances)-0.2*np.max(variances),0),arrowprops = dict(facecolor='black', shrink=0.05))
        #plt.annotate("border+", xy=(np.min(variances)+0.2*np.min(variances),0),arrowprops = dict(facecolor='black', shrink=0.05))
        #plt.xlabel('Neuron')
        #plt.ylabel('Var Value')
        plt.title('Var Values of Input Neurons')
        plt.savefig('MeanVarPlots/var_neuron_'+str(self.counter)+'.png')
        plt.close()


    def plotting_ACE(self):
        input_samples=self.new_inputs.detach().numpy()
        input_samples=input_samples.T
        for i,causal_effects in enumerate(np.array(self.input_samples_ACE).T):
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
        #print(len(self.input_samples_ACE))
        #print(len(self.input_samples_ACE[0]))
        print("DONE")
        #print(len(inputs))
        #print(len(inputs[0]))
        #check with sizes what we actually calculated

    def plotting_derivatives(self, first_orders_mean, high_orders_mean):
        plt.plot(first_orders_mean)
        plt.xlabel('Nr')
        plt.ylabel('Dev value')
        plt.title('first_orders_mean')
        plt.savefig('DEVplots/first_orders_mean'+str(self.counter)+'.png')
        plt.close()
        plt.plot(high_orders_mean)
        plt.xlabel('Nr')
        plt.ylabel('Dev value')
        plt.title('high_orders_mean')
        plt.savefig('DEVplots/high_orders_mean'+str(self.counter)+'.png')
        plt.close()
