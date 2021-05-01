import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt
import causal
import copy

class neural_network(object):
    """docstring for neural_network."""

    def __init__(self, learning_rate,net,criterion,opt,epochs,inputs_training,target_training,inputs_test,target_test,causality_on):
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

    def model(self,inputs):
        torch.set_default_dtype(torch.float64)
        dim = inputs.shape[1]
        self.net = nn.Sequential(
            #nn.Linear(dim, 50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.Sigmoid(),
            #nn.Linear(50,   1)

            #realNN
            nn.Linear(dim, 50, bias = True), nn.ReLU(),
            nn.Linear(50,   100, bias = True), nn.ReLU(),
            nn.Linear(100,   50, bias = True), nn.ReLU(),
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
        stepsave = []
        test_loss_training=[]
        for i in range(self.epochs):

            placeholderNet=copy.deepcopy(self.net)
            placeholderNet.eval()

            self.net.train()
            y_hat = self.net(x_train_t)
            loss = self.criterion(y_train_t,self.net(x_train_t))
            loss_training.append(loss.item())
            stepsave.append(i)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            #y_hat_class = (y_hat.detach().numpy())
            #accuracy = np.sum(self.y_train.reshape(-1,1)== y_hat_class )/len(self.target_training)
            self.net.eval()
            ypred = self.net(torch.from_numpy(self.inputs_test).detach())
            loss_test = self.criterion(torch.from_numpy(self.target_test).clone().reshape(-1, 1),ypred)
            test_loss_training.append(loss_test)
            if i > 0 and i % 10 == 0:
                print('Epoch %d, loss = %g' % (i, loss))



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

            if self.causality_on==1:
                self.update_weights_bias(placeholderNet)



        return loss_training,test_loss_training


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
        causal_test=causal.causality(self,0,0,0,0)
        causal_test.slicing_NN(self.inputs_training)
        return causal_test.final_causality

    def testing_training(self,loss_training,test_loss_training,figure_name):
        sl =np.array(loss_training)
        test=np.array(test_loss_training)

        plt.plot(sl[300:])
        plt.plot(test[300:])
        plt.xlabel('Actual value of training set')
        plt.ylabel('Prediction')
        plt.savefig(figure_name)

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
