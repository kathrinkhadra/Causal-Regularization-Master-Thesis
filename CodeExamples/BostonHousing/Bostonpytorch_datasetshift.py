# coding: utf-8
# based on https://discuss.pytorch.org/t/pytorch-fails-to-over-fit-boston-housing-dataset/40365

#rmse for test 7.52719

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

boston = load_boston()
X,y   = (boston.data, boston.target)
dim = X.shape[1]


X.shape

y



# Skip the next four lines if BostonHousing.csv is not available.
#house = pd.read_csv('BostonHousing.csv')
#print(house.head(10))
#house.hist(column='medv', bins=50)
#plt.show()
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = (y - y.min()) / (y.max() - y.min())

sample_sizes=X.shape[0]
#print(round(sample_sizes*0.1))

sample_sizes=y.shape[0]
#print(round(sample_sizes*0.1))
indice=round(sample_sizes*0.1)-1
#print('---------------------')
#A=np.array_split(X, round(sample_sizes*0.1))
X_train=X[:indice,:]
#print(X_train.shape)
X_test=X[indice:,:]
#print(X_test.shape)
#print('---------------------')
y_train=y[:indice]
#print(y_train.shape)
y_test=y[indice:]
#print(y_test.shape)
#print('---------------------')
performance= []
#for rand in range(370,400):
#    print(rand)
rand=369
finalindices=sample_sizes-indice
#rand=363#random.randint(350, 400)#random.randint(indice, finalindices)
#print(rand)

X_train=X[rand:indice+rand,:]
#print(X_train.shape)
X_test=np.concatenate((X[:rand,:],X[indice+rand:,:]))
#print(X_test.shape)
#print('---------------------')
y_train=y[rand:indice+rand]
#print(y_train.shape)
y_test=np.concatenate((y[:rand],y[indice+rand:]))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)
num_train = X_train.shape





# Scale both the training inputs and outputs
#X_train= scaler.fit_transform(X_train) #scaled_train
#X_test= scaler.fit_transform(X_test)

torch.set_default_dtype(torch.float64)
net = nn.Sequential(
    #nn.Linear(dim, 50, bias = True), nn.ELU(),
    #nn.Linear(50,   50, bias = True), nn.ELU(),
    #nn.Linear(50,   50, bias = True), nn.Sigmoid(),
    #nn.Linear(50,   1)
    nn.Linear(dim, 50, bias = True), nn.ReLU(),
    nn.Linear(50,   100, bias = True), nn.ReLU(),
    nn.Linear(100,   50, bias = True), nn.ReLU(),
    nn.Linear(50,   1)
)
criterion = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr = .0005)


num_epochs = 450
#from torch.utils.data import TensorDataset, DataLoader
y_train_t =torch.from_numpy(y_train).clone().reshape(-1, 1)
x_train_t =torch.from_numpy(X_train).clone()
#dataset = TensorDataset(torch.from_numpy(X_train).detach().clone(), torch.from_numpy(y_train).reshape(-1,1).detach().clone())
#loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
losssave = []
stepsave = []
test_losssave=[]
for i in range(num_epochs):
    net.train()
    y_hat = net(x_train_t)
    loss = criterion(y_train_t,net(x_train_t))
    losssave.append(loss.item())
    stepsave.append(i)
    loss.backward()
    opt.step()
    opt.zero_grad()
    y_hat_class = (y_hat.detach().numpy())
    accuracy = np.sum(y_train.reshape(-1,1)== y_hat_class )/len(y_train)
    net.eval()
    ypred = net(torch.from_numpy(X_test).detach())
    loss_test = criterion(torch.from_numpy(y_test).clone().reshape(-1, 1),ypred)
    test_losssave.append(loss_test)
    if i > 0 and i % 100 == 0:
        print('Epoch %d, loss = %g acc = %g ' % (i, loss,  accuracy))


ss=np.array(stepsave)
ss.shape
sl =np.array(losssave)
sl.shape
test=np.array(test_losssave)
#print (y_hat_class)
#print(y_train.reshape(-1,1))
#ss.reshape(8000)
#sl.reshape(8000)

#py = net(torch.DoubleTensor(X_train))
plt.plot(sl[300:])#'+'
plt.plot(test[300:])
plt.xlabel('Actual value of training set')
plt.ylabel('Prediction')
plt.show()

net.eval()
ypred = net(torch.from_numpy(X_test).detach())
loss_test = criterion(torch.from_numpy(y_test).clone().reshape(-1, 1),ypred)
print(loss_test)
#print('---------------')
#step_size=30
#beginning=200
#for i in range(beginning,len(test)-step_size,step_size):
#    print(i+np.argmin(test[i:i+step_size]))
#    print(np.min(test[i:i+step_size]))

#err = ypred.detach().numpy() - y_test
#mse = np.mean(err*err)
#print(mse)
#plt.plot(ypred.detach().numpy(),y_test, '+')
#plt.show()


#model = MLPRegressor(
#    hidden_layer_sizes=(50,50,50),
#    alpha = 0,
#    activation='relu',
#    batch_size=128,
#    learning_rate_init = 1e-3,
#    solver = 'adam',
#    learning_rate = 'constant',
#    verbose = False,
#    n_iter_no_change = 1000,
#    validation_fraction = 0.0,
#    max_iter=1000)
#model.fit(X_train, y_train)

#py = model.predict(X_test)
#err = y_test - py
#mse = np.mean(err**2)
#rmse = np.sqrt(mse)
#print('rmse for test %g' % rmse)
#plt.subplot(121)
#plt.plot(y_test, py, '+')
#plt.show()
#err = y_train - model.predict(X_train)
#mse = np.mean(err**2)

#plt.plot(py)
#py.mean()
