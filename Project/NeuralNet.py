
import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import causal
import copy

class neural_network(object):
    """docstring for neural_network."""

    def __init__(self, learning_rate,net,criterion,opt,epochs,inputs_training,target_training,inputs_test,target_test,causality_on,txt_name,epoch,factor,ACE_value,variance):
        super(neural_network, self).__init__()
        self.learning_rate = learning_rate
        self.net=net
        self.criterion=criterion
        self.opt=opt
        self.epochs=epochs
        self.inputs_training=inputs_training
        self.target_training=target_training
        self.inputs_test=inputs_test
        self.target_test=target_test
        self.causality_on=causality_on
        self.txt_name=txt_name
        self.epoch=epoch
        self.factor=factor
        self.ACE_value=ACE_value
        self.variance=variance

    def model(self,inputs):
        torch.set_default_dtype(torch.float64)
        dim = inputs.shape[1]
        self.net = nn.Sequential(
            #nn.Linear(dim, 50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.Sigmoid(),
            #nn.Linear(50,   1)

            #realNN
            nn.Linear(dim, 50, bias = True), nn.Sigmoid(),#nn.ReLU(),
            nn.Linear(50,   100, bias = True), nn.Sigmoid(),#nn.ReLU(),
            nn.Linear(100,   50, bias = True), nn.Sigmoid(),#nn.ReLU(),
            nn.Linear(50,   1)

            #nn.Linear(dim, 50, bias = True), nn.ReLU(),
            #nn.Linear(50,   50, bias = True), nn.ReLU(),
            #nn.Linear(50,   50, bias = True), nn.ReLU(),
            #nn.Linear(50,   50, bias = True), nn.ReLU(),
            #nn.Linear(50,   10, bias = True), nn.ReLU(),
            #nn.Linear(10,   1)

            #nn.Linear(dim, 50, bias = True), nn.Sigmoid(),
            #nn.Linear(50,   50, bias = True), nn.Tanh(),
            #nn.Linear(50,   10, bias = True), nn.Softmax(),
            #nn.Linear(10,   50, bias = True), nn.ReLU(),
            #nn.Linear(50,   10, bias = True), nn.Sigmoid(),
            #nn.Linear(10,   1)
        )
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

    def training(self):
        y_train_t =torch.from_numpy(self.target_training).clone().reshape(-1, 1)
        x_train_t =torch.from_numpy(self.inputs_training).clone()
        #dataset = TensorDataset(torch.from_numpy(inputs_training).detach().clone(), torch.from_numpy(y_train).reshape(-1,1).detach().clone())
        #loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        loss_training = []
        loss_control_training_MSE=[]
        stepsave = []
        test_loss_training=[]
        ACE_values=[]
        variances=[]
        for i in range(self.epochs):
            self.epoch=i

            placeholderNet=copy.deepcopy(self.net)
            placeholderNet.eval()

            self.net.train()
            y_hat = self.net(x_train_t)
            #loss = self.criterion(y_train_t,self.net(x_train_t))
            if self.causality_on==1:
                loss=self.my_loss(y_train_t,self.net(x_train_t))
            else:
                loss = self.criterion(y_train_t,self.net(x_train_t))
            self.net.train()
            loss_training.append(loss.item())
            stepsave.append(i)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            #y_hat_class = (y_hat.detach().numpy())
            #accuracy = np.sum(self.y_train.reshape(-1,1)== y_hat_class )/len(self.target_training)
            self.net.eval()
            ypred = self.net(torch.from_numpy(self.inputs_test).detach())
            loss_test = self.criterion(torch.from_numpy(self.target_test).clone().reshape(-1, 1),ypred)
            test_loss_training.append(loss_test.item())
            #print("one done")
            #if i > 0 and i % 10 == 0:
            print('Epoch %d, loss = %g' % (i, loss))

            if self.causality_on==1:
                loss_control_training_MSE.append(self.criterion(y_train_t,self.net(x_train_t)).item())



            #print("weight experiment")

            #print(len(self.net[0].weight.grad))
            #print(len(self.net[0].weight.grad[0]))
            #print(len(self.net.bias.grad[0]))
            #print(np.array(self.net[0].bias.grad).shape)

            #print(len(self.net[6].weight.grad))
            #print(len(self.net[6].weight.grad[0]))
            #print(len(self.net.bias.grad[0]))
            #print(np.array(self.net[6].bias.grad).shape)

            #self.net[0].weight=torch.nn.Parameter(torch.from_numpy(np.array(0)))
            #print(len(self.net[2].weight[0]))
            #print(self.net[0].weight.detach().numpy().shape)
            if self.causality_on==0:
                self.net.eval()
                mean=self.ACE_regularitzation(y_train_t,self.net(x_train_t))
                self.ACE_value=-mean

            if i % 10 == 0:
                ACE_values.append(np.array(self.ACE_value))
                variances.append(np.array(self.variance))
                #print(ACE_values)
                #print(np.mean(variances))
            #if self.causality_on==1:
            #    self.update_weights_bias(placeholderNet)
            #print(loss_training)
            #print(test_loss_training)
            #print(loss_control_training_MSE)
        f = open(self.txt_name, 'a')
        f.write('loss_training='+str(loss_training)+'\n\n')
        f.write('test_loss_training='+str(test_loss_training)+'\n\n')
        f.write('variances='+str(variances)+'\n\n')
        f.write('loss_control_training_MSE='+str(loss_control_training_MSE)+'\n\n')
        f.write('variances='+str(np.mean(variances))+'\n\n')
        f.write('ACE_values='+str(ACE_values)+'\n\n')

        return loss_training,test_loss_training

    def my_loss(self,target,output):
        #a=self.criterion(target,output)
        #print(self.criterion(target,output)) #50 samples = 50 a
        #value=self.ACE_regularitzation(target,output)
        loss = self.criterion(target,output) + self.factor*self.ACE_regularitzation(target,output)#torch.tensor(self.ACE_regularitzation(target,output))

        return loss

    def ACE_regularitzation(self,target,output):
        mean,self.variance = self.ACE_function()
        #mean=np.concatenate(mean, axis=None)
        self.ACE_value=torch.mean(mean)
        value=-torch.mean(mean)
        return value

    def ACE_function(self):
        causal_regularization=causal.causality(self,0,0,0,0,0,0,self.txt_name)
        causal_regularization.slicing_NN(self.inputs_training,self.epoch)
        return causal_regularization.means, causal_regularization.variances

    def update_weights_bias(self,placeholderNet):
        causal_binary = self.ACE_execution()
        #self.net[2].weight
        for indx,binary in enumerate(causal_binary):
            #i=indx*2
            #print(i)
            #print(indx)
            new_bias=self.selection_bias(placeholderNet,indx,binary)
            new_weights=self.selection_weights(placeholderNet,indx,binary)
            self.overwrite_weights_bias(placeholderNet,indx,new_weights,new_bias)

            #print("shape weights new")
            #print(self.net[indx*2].weight.detach().numpy().shape)
            #print("shape weights update")
            #print(weight_update.shape)
            #print("shape weights original")
            #print(placeholderNet[indx*2].weight.detach().numpy().shape)
            #print("shape weights keep")
            #print(weighted_keep.shape)
            #print("shape weights final")
            #print(new_weights.shape)

    def selection_bias(self,placeholderNet,indx,binary):

        bias_update=np.multiply(self.net[indx*2].bias.detach().numpy(), np.array(binary))
        inverted_binary=1-np.array(binary)
        bias_keep=np.multiply(placeholderNet[indx*2].bias.detach().numpy(), inverted_binary)

        new_bias=bias_update+bias_keep

        #print("shape bias net")
        #print(self.net[indx*2].bias.detach().numpy().shape)
        #print("bias_update")
        #print(bias_update.shape)
        #print("new_bias")
        #print(new_bias.shape)
        #print("bias net")
        #print(self.net[indx*2].bias.detach().numpy())
        #print("binary")
        #print(binary)
        #print("bias_update")
        #print(bias_update)

        #print("bias placeholderNet")
        #print(placeholderNet[indx*2].bias.detach().numpy())
        #print("new_bias")
        #print(new_bias)

        return new_bias


    def selection_weights(self,placeholderNet,indx,binary):

        weight_update=np.multiply(self.net[indx*2].weight.detach().numpy(), np.array(binary)[:, np.newaxis])
        inverted_binary=1-np.array(binary)
        weighted_keep=np.multiply(placeholderNet[indx*2].weight.detach().numpy(), inverted_binary[:, np.newaxis])

        new_weights=weight_update+weighted_keep

        #print("shape weights net")
        #print(self.net[indx*2].weight.detach().numpy().shape)
        #print("weight_update")
        #print(weight_update.shape)
        #print("new_weights")
        #print(new_weights.shape)
        #print("weight net")
        #print(self.net[indx*2].weight.detach().numpy())
        #print("binary")
        #print(binary)
        #print("weight_update")
        #print(weight_update)
        #print("new_weights")
        #print(new_weights)

        return new_weights


    def overwrite_weights_bias(self,placeholderNet,indx,new_weights,new_bias):
        self.net[indx*2].weight=torch.nn.Parameter(torch.from_numpy(new_weights))
        self.net[indx*2].bias=torch.nn.Parameter(torch.from_numpy(new_bias))

    def ACE_execution(self):
        causal_test=causal.causality(self,0,0,0,0,0,0,self.txt_name)
        causal_test.slicing_NN(self.inputs_training,self.epoch)
        return causal_test.final_causality


    def testing_training(self,loss_training,test_loss_training,figure_name):
        sl =np.array(loss_training)
        test=np.array(test_loss_training)

        plt.plot(sl[300:], label="training loss")
        plt.plot(test[300:], label="validation loss")
        plt.xlabel('Actual value of training set')
        plt.ylabel('Prediction')
        plt.legend(loc='upper right')
        plt.savefig(figure_name)
        plt.close()

    def testing(self):
        self.net.eval()
        ypred = self.net(torch.from_numpy(self.inputs_test).detach())
        print(ypred.shape)
        print(self.net)
        loss_test = self.criterion(torch.from_numpy(self.target_test).clone().reshape(-1, 1),ypred)
        print(loss_test)

        return loss_test

    #def extract_activations(self):
    #    print(1)
