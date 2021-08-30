from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
