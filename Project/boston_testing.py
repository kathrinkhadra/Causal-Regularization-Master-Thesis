from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

boston = load_boston()
inputs,target   = (boston.data, boston.target)
scaler = MinMaxScaler(feature_range=(0, 1))
inputs = scaler.fit_transform(inputs)
target  = (target  - target.min()) / (target.max() - target.min())


#print(inputs.shape)

#for feature in inputs.T:
#    print("feature")
#    print(feature.shape)
#    feature=np.array(list(set(feature)))
#    print(feature.shape)
#    u, c = np.unique(feature, return_counts=True)
    #dup = u[c > 1]
#    print(c)


values=[]
indices=[]

for indx in range(inputs.shape[1]):
    val,indi=np.unique(inputs[:,indx], return_index=True)
    values.append(val)
    indices.append(np.array(indi))
    #print(indi)
    #val,c=np.unique(inputs[:,indx], return_counts=True)
    #dup=val[c > 1]
#indices=np.array(indices).T

print(indices[0])
#print(indices.shape)
#print(indices.T.shape)

print(indices[6])


tensor=[torch.tensor([ 0.0001,  0.0016,  0.0016, -0.0078,  0.0005, -0.0010,  0.0013, -0.0006,
         0.0016,  0.0016,  0.0016,  0.0006,  0.0003]), torch.tensor([-7.5046e-05, -8.3345e-05, -7.5046e-05, -7.3819e-05, -6.0660e-05,
        -7.5046e-05, -5.4562e-05, -7.5046e-05, -7.5046e-05, -1.6737e-04,
        -3.5970e-05, -7.5046e-05,  5.9404e-04, -7.5046e-05, -5.5421e-05,
         4.4345e-05,  6.6016e-05,  4.8033e-04, -7.5046e-05, -7.5046e-05,
         3.0140e-05,  5.2797e-05,  1.4439e-04, -3.0594e-04, -7.5046e-05,
         4.8886e-07, -1.8477e-06, -7.5046e-05,  6.1192e-05, -7.5046e-05,
        -1.4330e-04, -7.5046e-05,  5.8080e-04, -7.5046e-05,  1.0366e-05,
        -7.5046e-05, -7.5046e-05, -4.9470e-04, -7.5046e-05, -7.5046e-05,
        -4.4794e-05, -1.0803e-04,  1.9047e-04, -7.2046e-04, -3.3669e-06,
        -7.5046e-05, -7.5046e-05, -2.4504e-05, -2.5309e-04,  3.8286e-05])]

tensor=torch.cat(tensor.unsqueeze(0),dim=0)

print(tensor)

#print(np.nan-50)





#def unique_values(self):
#    values=[]
#    indices=[]
#    for indx in range(self.new_inputs.shape[1]):
#        val,indi=np.unique(self.new_inputs[:,indx], return_index=True)
#        values.append(np.array(val))
#        indices.append(np.array(indi))
#    return np.array(values),np.array(indices).T


#if index_input in indices[indx]:
#    continue
#    print("True")

#for index_input, input_sample in enumerate(self.new_inputs):
