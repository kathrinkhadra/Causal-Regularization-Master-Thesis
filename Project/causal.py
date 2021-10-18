import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import copy

class causality(object):
    """docstring for causality."""

    def __init__(self, neural_network,new_inputs,input_samples_ACE,counter,final_causality,mean, variances,txt_name):
        super(causality, self).__init__()
        self.neural_network = neural_network
        self.new_inputs=new_inputs
        self.input_samples_ACE=input_samples_ACE
        self.counter=counter
        self.final_causality=final_causality
        self.means=mean
        self.variances=variances
        self.txt_name=txt_name

    def cov(self,tensor, rowvar=True, bias=False):
        """Estimate a covariance matrix (np.cov)"""
        tensor = tensor if rowvar else tensor.transpose(-1, -2)
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
        return factor * tensor @ tensor.transpose(-1, -2).conj()

    def slicing_NN(self,input_sample,iteration):
        #slicing after the unlinearity needs to be done
        #y_train_t =torch.from_numpy(self.target_training).clone().reshape(-1, 1)
        #print(input_sample.shape)
        self.txt_name=self.txt_name.replace(".txt", "")
        self.txt_name=self.txt_name+'_ACEvalues.txt'
        self.final_causality=[]
        self.means=[]
        self.variances=[]
        x_train_t = input_sample#torch.from_numpy(input_sample).clone()
        for counter,layer in enumerate(self.neural_network.net):
            #self.new_inputs=torch.tensor(0.0)
            #self.new_inputs.requires_grad=True
            if (counter%2==1 and counter<len(self.neural_network.net)-1) or counter==0:#FOR DROPOUT#counter%2==1 and counter<len(self.neural_network.net)-1 and counter!=0:#
                self.counter=counter
                #print(counter)
                #self.neural_network.net.train()
                #self.neural_network.net.eval()
                new_nn=self.neural_network.net[counter:len(self.neural_network.net)]
                placeholder_nn=self.neural_network.net[0:counter]
                #print(new_nn.parameters.requires_grad)
                #new_nn.eval()
                #self.new_inputs=torch.tensor(self.neural_network.net[0:counter](x_train_t), requires_grad=True)#create_graph=True)#
                self.new_inputs=placeholder_nn(x_train_t)
                #print(placeholder_nn)
                #print(self.new_inputs)
                #self.neural_network.net.train()
                #new_nn.train()
                #print(self.new_inputs)
                #covariance=torch.tensor(np.cov(self.new_inputs.detach().numpy(),rowvar=False))
                covariance=self.cov(self.new_inputs,rowvar=False)
                #print(covariance)
                #covariance=torch.cov(self.new_inputs,rowvar=False)
                #mean=np.array(np.mean(self.new_inputs.detach().numpy(), axis=0))#double check axis maybe more a axis of 1
                #print(mean)
                ##print(counter)
                #self.new_inputs.requires_grad=True
                #self.new_inputs=torch.tensor(self.new_inputs.data, requires_grad=True)
                mean=torch.mean(self.new_inputs, axis=0)
                #print(mean.requires_grad)
                #mean.requires_grad=True
                #mean.requires_grad=True
                #print("mean shape")
                #print(mean)
                #print(covariance)
                #shapes are wrong############################
                #print(self.new_inputs.detach().numpy().shape)

                #print("nn generating input")
                #print(self.neural_network.net[0:counter])
                #print("sliced NN")
                #print(new_nn)

                self.ACE(covariance,mean,new_nn)

                causality_update, medians, mean_variances=self.evaluating_ACE()

                self.final_causality.append(causality_update)
                self.means.append(medians)
                self.variances.append(mean_variances)

                #self.final_causality.append(causality_update, dtype=object)
                #if counter==0:
                #self.plotting_ACE()
                #print("self.new_inputs shape")
                #print(self.new_inputs.detach().numpy().shape)
                #print(mean)
                #print(covariance)
        #print(self.means)
        #print(self.variances)
        #self.final_causality=torch.cat(self.final_causality)
        #print(self.variances)
        self.means=torch.cat(self.means,dim=0)
        #print(self.means)
        self.variances=torch.var(torch.cat(self.variances,dim=0))
        #print(self.variances)

        if iteration% 10 == 0:
            f = open(self.txt_name, 'a')
            f.write('-----------------------------------------------ITERATION'+str(iteration)+'-----------------------------------------------\n\n')
            f.write('final_causality='+str(self.final_causality)+'\n\n')
            f.write('means='+str(self.means)+'\n\n')
            f.write('mean_overall='+str(torch.mean(self.means))+'\n\n')

            #f.write('variances='+str(self.variances)+'\n\n')
            #f.write('-----------------------------------------------ITERATION-----------------------------------------------\n\n')

    def unique_values(self):
        values=[]
        indices=[]
        #print(self.new_inputs.shape[1])
        for indx in range(self.new_inputs.shape[1]):
            #if indx==1:
            #    print(self.new_inputs.detach().numpy()[:,indx])
            #val,indi=np.unique(self.new_inputs.detach().numpy()[:,indx], return_index=True)
            #values.append(np.array(val))
            #indices.append(np.array(indi))
            val,inv = torch.unique(self.new_inputs[:,indx], sorted=True, return_inverse=True)
            indi = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            inv, indi = inv.flip([0]), indi.flip([0])
            indi = inv.new_empty(val.size(0)).scatter_(0, inv, indi)
            values.append(val)
            indices.append(indi)
            #print(len(indices))
        #print(indices)
        return values,indices#np.array(values,dtype=object),indices#.T

    def get_input_samples(self):
        #inv_value=[]
        #for ind,min in enumerate(torch.min(self.new_inputs, 0).values):
        #    print(ind)
        #    print(min)
        #    print(torch.max(self.new_inputs, 0).values[ind])
        #    inv_value.append(torch.linspace(min.data,torch.max(self.new_inputs, 0).values[ind].data,50))

        #inv_value=torch.stack(inv_value)
        number_of_inv=10
        inv_value=torch.stack([torch.linspace(min.data,torch.max(self.new_inputs, 0).values[ind].data,number_of_inv) for ind,min in enumerate(torch.min(self.new_inputs, 0).values)])
        #print(inv_value)
        return torch.transpose(inv_value,0,1)

    def ACE(self,covariance,mean,neural_net):
        torch.set_default_dtype(torch.float64)
        self.input_samples_ACE=[]
        first_orders_mean_array=[]
        high_orders_mean_array=[]
        inv_values=self.get_input_samples()
        #print(inv_values[:,1])
        #print(len(inv_values[0]))
        #print(self.new_inputs[:,1])
        #print(len(self.new_inputs[0]))
        #unique_values, unique_indices=self.unique_values()
        #print("size unique_indices")
        #print(len(unique_indices))
        #print(unique_indices)

        store_ACEs=[]
        inv_value_counter=0
        for index_input, input_sample in enumerate(inv_values):#self.new_inputs

            first_orders_mean=[]
            high_orders_mean=[]
            #input_sample=input_sample.detach().numpy()
            #print(input_sample)
            average_causal_effects=[]
            for indx in range(len(input_sample)):
                #print(len(input_sample))

                #if index_input not in unique_indices[indx]:
                    #print("True")
                #    average_causal_effects.append(float("nan"))
                #    continue





                expectation_do_x = []
                mean_vector=mean.clone()#.detach()
                #print(mean_vector)
                #print(input_sample)
                mean_vector[indx] = input_sample[indx]#problem 2D = new input and mean vector is not supposed to be
                input_tensor=mean_vector.clone()#.detach()
                #print(input_tensor.requires_grad)

                output=neural_net(input_tensor) # torch.from_numpy(mean_vector)
                #print(output)
                #print(neural_net)

                #output = torch.nn.functional.sigmoid(output)
                #output = torch.nn.functional.softmax(output)
                #print(output[0].data)
                val = output#output.data#output.data.view(1).cpu().numpy()[0]
                #print(output.data.view(1).cpu().numpy()[0])
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
                #first_orders_mean.append(torch.mean(first_grads[0].data))#np.array(
                for dim in range(len(mean)):
                    #print(len(mean))
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
                    #higher_order_grads_array = higher_order_grads[0].data#np.array()
                    #print(higher_order_grads)
                    high_orders_mean.append(torch.mean(higher_order_grads[0]))#_array
                    #print(high_orders_mean)

                    temp_cov = covariance.clone()#.detach()
                    temp_cov[dim][indx] = 0.0
                    #print(temp_cov)
                    val += 0.5*torch.sum(higher_order_grads[0]*temp_cov[dim])
                    #print(val)

                average_causal_effects.append(val)


            #average_causal_effects = np.array(average_causal_effects) - np.mean(np.array(average_causal_effects))
                first_orders_mean_array.append(first_orders_mean)
                high_orders_mean_array.append(high_orders_mean)
                #print("FIRST LOOP")
                #print(len(average_causal_effects))
                #print(len(average_causal_effects[0]))
            #average_causal_effects=np.array(average_causal_effects)
            average_causal_effects=torch.stack(average_causal_effects)
            #print("average_causal_effects")
            #print(len(average_causal_effects))
            #print(len(average_causal_effects[0]))

            #print(average_causal_effects)
            #print(average_causal_effects[~torch.any(average_causal_effects.isnan(),dim=0)])
            #print(average_causal_effects[~torch.isnan(average_causal_effects)])
            #print(average_causal_effects[~np.isnan(average_causal_effects)])
            #print(np.mean(average_causal_effects[~np.isnan(average_causal_effects)]))
            #print("--------------------")

            #print(average_causal_effects)

            store_ACEs.append(average_causal_effects)

            #[~torch.isnan(average_causal_effects)]#average_causal_effects[~np.isnan(average_causal_effects)]
            #
        #print(inv_value_counter)
        #print("store_ACEs")
        #print(len(store_ACEs))
        #print(len(store_ACEs[0]))
        #print("store_ACEs")
        #print(store_ACEs)
        store_ACEs=torch.squeeze(torch.stack(store_ACEs)).T
        #print("store_ACEs")
        #print(store_ACEs.shape)
        #print(len(store_ACEs[0]))
        means=torch.mean(store_ACEs,1)
        #print("means")
        #print(len(means))
        #print(len(means[0]))
        #print(store_ACEs)
        #print(len(store_ACEs))
        #print(len(store_ACEs[0]))
        #print(means)
        self.input_samples_ACE=[ace - means[indx] for indx, ace in enumerate(store_ACEs)]
        #print("input_samples_ACE")
        #print(len(self.input_samples_ACE))
        #print(len(self.input_samples_ACE[0]))

        #print(self.input_samples_ACE)
        #for indx,ace in enumerate(store_ACEs):
        #    print("ace")
        #    print(ace)
        #    print("mean")
        #    print(means[indx])
        #    print(ace - means[indx])
        #    self.input_samples_ACE.append()
        #print(self.input_samples_ACE)
        #print(len(self.input_samples_ACE))
        #print(len(self.input_samples_ACE[0]))
        #print(average_causal_effects)

            #self.plotting_derivatives(first_orders_mean_array, high_orders_mean_array)
        #ACE_txt_name=self.txt_name
        #ACE_txt_name=ACE_txt_name.replace("_ACEvalues.txt", "")
        #ACE_txt_name=ACE_txt_name+'_DETAILEDVALUES.txt'
        #f = open(ACE_txt_name, 'a')
        #f.write('input_samples_ACE_'+str(self.counter)+'='+str(self.input_samples_ACE)+'\n\n')

    def evaluating_ACE(self):
        #print(torch.stack(self.input_samples_ACE))
        #print(len(self.input_samples_ACE[2]))
        #print(len(self.input_samples_ACE[8]))
        ACEs=torch.stack(self.input_samples_ACE)#.T
        #print(ACEs)
        #print("ACEs")
        #print(len(ACEs))
        #print(len(ACEs[0]))


        #ACEs=[np.array(sample[~np.isnan(sample)],dtype=float) for sample in ACEs]#sample[~np.isnan(sample)]]
        #print(ACEs)
        #ACEs=np.array(ACEs)
        #ACEs=np.concatenate(ACEs, axis=1)
        #ACEs=ACEs[:, ~np.isnan(ACEs).any(axis=0)]
        #ACEs = np.ma.array(ACEs, mask=np.isnan(ACEs))
        #print(ACEs)
        #print(ACEs.shape)
        #for ace in ACEs:
        #    print(ace[~torch.isnan(ace)])
        #    print(torch.median(ace[~torch.isnan(ace)]))
        medians=[torch.absolute(torch.median(ace)) for ace in ACEs]#[~torch.isnan(ace)]
        medians=torch.stack(medians)
        #print()
        variances=[torch.var(ace) for ace in ACEs]#np.var(ACEs, axis=1)#[~torch.isnan(ace)]
        variances=torch.stack(variances)

        variances_mean=[torch.median(ace) for ace in ACEs]
        variances_mean=torch.stack(variances_mean)
        #print(medians)
        variances[variances != variances]=0
        #print(variances)
        #print(medians.data.size())
        #print(variances.data.size())
        #print("--------------------ACEshape-------------------------")
        #print(ACEs.shape)
        #plus_border_mean=np.percentile(medians,80)
        #minus_border_mean=np.percentile(medians,20)
        #plus_border_variances=np.percentile(variances,80)
        #minus_border_variances=np.percentile(variances,20)
        #np.max(medians)
        #np.min(medians)


        #medians[medians>=plus_border_mean]=0
        #medians[medians<plus_border_mean]=1
        #medians[medians<=minus_border_mean]=0
        #medians[medians>minus_border_mean]=1
        #print("get causality")
        #print(plus_border_mean)
        #print(minus_border_mean)
        #print(medians)


        #causality_update = [0 if med>=plus_border_mean or med<=minus_border_mean else 1 for med in medians]

        causality_update = [-1 if med<0 else 1 for med in medians]
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
        return causality_update, medians, variances_mean

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
        #input_samples=self.new_inputs.detach().numpy()
        #input_samples=input_samples.T
        for i,causal_effects in enumerate(np.array(self.input_samples_ACE).T):
            #print(i)
            if i>14:
                break
            #print(causal_effects.shape)
            #print()
            #plt.scatter(input_samples[i,:],causal_effects)
            #plt.xlabel('Interventional Value')
            #plt.ylabel('ACE')
            #plt.title('average_causal_effects_neuron_'+str(i))
            #plt.savefig('ACEplots/neuron'+str(i)+'.png')
            #plt.close()
            plt.hist(causal_effects)
            plt.xlabel('ACE')
            plt.ylabel('Häufigkeit')
            plt.title('histogram_average_causal_effects_neuron_'+str(i))
            plt.savefig('ACEplots/histogram_neuron'+str(i)+'_step_'+str(self.counter)+'.png')
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
