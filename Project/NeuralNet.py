import torch
import torch.nn as nn
import numpy  as np
import matplotlib.pyplot as plt

class neural_network(object):
    """docstring for neural_network."""

    def __init__(self, learning_rate,net,criterion,opt,epochs,inputs_training,target_training,inputs_test,target_test):
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

    def model(self,inputs):
        torch.set_default_dtype(torch.float64)
        dim = inputs.shape[1]
        self.net = nn.Sequential(
            #nn.Linear(dim, 50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.ELU(),
            #nn.Linear(50,   50, bias = True), nn.Sigmoid(),
            #nn.Linear(50,   1)
            nn.Linear(dim, 50, bias = True), nn.ReLU(),
            nn.Linear(50,   100, bias = True), nn.ReLU(),
            nn.Linear(100,   50, bias = True), nn.ReLU(),
            nn.Linear(50,   1)
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

        return loss_training,test_loss_training

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
