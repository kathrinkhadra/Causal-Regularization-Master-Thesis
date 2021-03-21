feature_name = {0:"SepalLength",1:"SepalWidth",2:"PetalLength",3:"PetalWidth",4:"Species"}
col={0:"b",1:"g",2:"r",3:"c"}
tit={0:"Iris-setosa",1:"Iris-versicolor",2:"Iris-virginica"}
for output_index in range(0,n_classes):#For every target
    plt.figure()
    for t in range(0,num_c):#For every feature/neuron etc
        expectation_do_x = []
        inp=copy.deepcopy(mean_vector)

        for x in np.linspace(0, 1, num_alpha):

            #model
            inp[t] = x
            input_torchvar = autograd.Variable(torch.FloatTensor(inp), requires_grad=True)

            output=F.softmax(model(input_torchvar), dim=-1)

            o1=output.data.cpu()
            val=o1.numpy()[output_index]#first term in interventional expectation

            grad_mask_gradient = torch.zeros(n_classes)
            grad_mask_gradient[output_index] = 1.0
            #calculating the hessian
            first_grads = torch.autograd.grad(output.cpu(), input_torchvar.cpu(), grad_outputs=grad_mask_gradient, retain_graph=True, create_graph=True)

            for dimension in range(0,num_c):#Tr(Hessian*Covariance)
                if dimension == t:
                  continue
                temp_cov = copy.deepcopy(cov) # cov of data X_values
                temp_cov[dimension][t] = 0.0#row,col in covariance corresponding to the intervened one made 0
                grad_mask_hessian = torch.zeros(num_c)
                grad_mask_hessian[dimension] = 1.0

                #calculating the hessian
                hessian = torch.autograd.grad(first_grads, input_torchvar, grad_outputs=grad_mask_hessian, retain_graph=True, create_graph=False)#gradient(gradient(x,y),y)

                val += np.sum(0.5*hessian[0].data.numpy()*temp_cov[dimension])#adding second term in interventional expectation: 0.5*Tr(delta^2* f'_y(mu)*inv.Cov.)
            expectation_do_x.append(val)#append interventional expectation for given interventional value
        plt.title(tit[output_index])
        plt.xlabel('Intervention Value(alpha)')
        plt.ylabel('Causal Attributions(ACE)')

        #Baseline is np.mean(expectation_do_x)
        plt.plot(np.linspace(0, 1, num_alpha), np.array(expectation_do_x) - np.mean(np.array(expectation_do_x)), label = feature_name[t],color=col[t])
        plt.legend()
        #Plotting vertical lines to indicate regions
        if output_index == 0:
            plt.plot(np.array([0.2916666567325592]*1000),np.linspace(-3,3,1000),"--")

        if output_index == 1:
            plt.plot(np.array([0.2916666567325592]*1000),np.linspace(-3,3,1000),"--")
            plt.plot(np.array([0.6874999403953552]*1000),np.linspace(-3,3,1000),"--")
        if output_index == 2:
            plt.plot(np.array([0.6874999403953552]*1000),np.linspace(-3,3,1000),"--")
        plt.savefig("/gdrive/My Drive/IRIS/Code/"+str(output_index)+".png")
