import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy  as np


class Dataprep(object):
    """docstring for Dataprep."""

    def __init__(self, inputs,target,inputs_training,inputs_test,target_training,target_test,test_size):
        super(Dataprep, self).__init__()
        self.inputs = inputs
        self.target=target
        self.inputs_training=inputs_training
        self.inputs_test=inputs_test
        self.target_training=target_training
        self.target_test=target_test
        self.test_size=test_size

    def data_loading(self):
        boston = load_boston()
        self.inputs,self.target   = (boston.data, boston.target)

    def data_preprocessing(self):

        self.data_loading()
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.inputs = scaler.fit_transform(self.inputs)
        self.target  = (self.target  - self.target.min()) / (self.target.max() - self.target.min())

    def splitting_data_noshift(self):

        self.data_preprocessing()
        self.inputs_training, self.inputs_test, self.target_training, self.target_test = train_test_split(self.inputs, self.target , test_size=self.test_size, random_state=0)

    def dataset_shift(self,rand):

        self.data_preprocessing()
        sample_sizes=self.inputs.shape[0]
        sample_sizes=self.target.shape[0]

        indice=round(sample_sizes*0.1)-1
        #rand=369#400,432
        finalindices=sample_sizes-indice

        self.inputs_training=self.inputs[rand:indice+rand,:]
        self.inputs_test=np.concatenate((self.inputs[:rand,:],self.inputs[indice+rand:,:]))

        self.target_training=self.target[rand:indice+rand]
        self.target_test=np.concatenate((self.target[:rand],self.target[indice+rand:]))
