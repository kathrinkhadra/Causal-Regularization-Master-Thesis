from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch


def cov_old(m, rowvar=False, inplace=False):
    '''Estimate a covariance matrix given data.

    Thanks :
    - https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    - https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py#L2276-L2494

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m.type(torch.double)
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else :
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def cov_old_old(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def main():

    boston = load_boston()
    inputs,target   = (boston.data, boston.target)
    scaler = MinMaxScaler(feature_range=(0, 1))
    inputs = scaler.fit_transform(inputs)
    target  = (target  - target.min()) / (target.max() - target.min())

    inputs_training, inputs_test, target_training, target_test = train_test_split(inputs, target , test_size=0.3, random_state=0)

    c=cov(torch.tensor(inputs_training), rowvar=False)
    a=np.cov(inputs_training, rowvar=False)

    #print(c)
    #print(a)

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
        #val,indi=torch.unique(inputs[:,indx])#, return_index=True)
        val,indi=np.unique(inputs[:,indx], return_index=True)
        values.append(val)
        indices.append(np.array(indi))
        #print(indi)
        #val,c=np.unique(inputs[:,indx], return_counts=True)
        #dup=val[c > 1]
    #indices=np.array(indices).T

    #print(indices[0])

    inputs=torch.tensor(inputs_training)
    for indx in range(inputs.shape[1]):
        #val,indi=torch.unique(inputs[:,indx])#, return_index=True)


        val,inv = torch.unique(inputs[:,indx], sorted=True, return_inverse=True)
        indi = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        inv, indi = inv.flip([0]), indi.flip([0])
        indi = inv.new_empty(val.size(0)).scatter_(0, inv, indi)
        #val,indi=np.unique(inputs[:,indx], return_index=True)
        values.append(val)
        indices.append(np.array(indi))

    #print(indices[0])
    #print(indices.shape)
    #print(indices.T.shape)

    a=torch.tensor([[-1.2360, -0.2942, -0.1222,  0.8475],[ 1.1949, -1.1127, -2.2379, -0.6702],[ 1.5717, -0.9207,  0.1297, -1.8768],[-0.6172,  1.0036, -0.6060, -0.2432]])
    #print(torch.max(a, 1))
    print(torch.max(a, 0).values)
    print(torch.min(a, 0).values)

    feature=[torch.linspace(min,torch.max(a, 0).values[ind],50) for ind,min in enumerate(torch.min(a, 0).values)]
    feature=torch.stack(feature)
    print(feature)
    #print(len(feature))
    feature=torch.transpose(feature,0,1)
    print(feature)
if __name__ == "__main__":
    main()

    #tensor=[torch.tensor([ 0.0001,  0.0016,  0.0016, -0.0078,  0.0005, -0.0010,  0.0013, -0.0006,
    #         0.0016,  0.0016,  0.0016,  0.0006,  0.0003]), torch.tensor([-7.5046e-05, -8.3345e-05, -7.5046e-05, -7.3819e-05, -6.0660e-05,
    #        -7.5046e-05, -5.4562e-05, -7.5046e-05, -7.5046e-05, -1.6737e-04,
    #        -3.5970e-05, -7.5046e-05,  5.9404e-04, -7.5046e-05, -5.5421e-05,
    #         4.4345e-05,  6.6016e-05,  4.8033e-04, -7.5046e-05, -7.5046e-05,
    #         3.0140e-05,  5.2797e-05,  1.4439e-04, -3.0594e-04, -7.5046e-05,
    #         4.8886e-07, -1.8477e-06, -7.5046e-05,  6.1192e-05, -7.5046e-05,
    #        -1.4330e-04, -7.5046e-05,  5.8080e-04, -7.5046e-05,  1.0366e-05,
    #        -7.5046e-05, -7.5046e-05, -4.9470e-04, -7.5046e-05, -7.5046e-05,
    #        -4.4794e-05, -1.0803e-04,  1.9047e-04, -7.2046e-04, -3.3669e-06,
    #        -7.5046e-05, -7.5046e-05, -2.4504e-05, -2.5309e-04,  3.8286e-05])]

    #tensor=torch.cat(tensor.unsqueeze(0),dim=0)

    #print(tensor)

    #print(np.nan-50)





    #def unique_values(self):
    #    values=[]
    #    indices=[]
    #    for indx in range(new_inputs.shape[1]):
    #        val,indi=np.unique(new_inputs[:,indx], return_index=True)
    #        values.append(np.array(val))
    #        indices.append(np.array(indi))
    #    return np.array(values),np.array(indices).T


    #if index_input in indices[indx]:
    #    continue
    #    print("True")

    #for index_input, input_sample in enumerate(new_inputs):
