##ACE##
lstm.zero_grad()
input_data = np.array(input_data)
average_causal_effects = []
for t in range(len(input_data)):
    expected_value = 0.0
    expectation_do_x = []
    inp = copy.deepcopy(mean_vector)

    inp[t] = input_data[t]
    hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True)
    input_torchvar = autograd.Variable(torch.FloatTensor(inp), requires_grad=True)
    output, hidden = lstm(input_torchvar.view(len(inp), 1, -1), hidden)
    output_2 = F.sigmoid(output_layer(output[-1]))

    val = output_2.data.view(1).cpu().numpy()[0]

    first_grads = torch.autograd.grad(output_2, input_torchvar, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False) #as only one output

    first_grad_shape = first_grads[0].data.size()
    lower_order_grads = first_grads
    
    for dimension in range(len(mean_vector)):
        if dimension == t:
            continue
        grad_mask = torch.zeros(first_grad_shape)
        grad_mask[dimension] = 1.0


        higher_order_grads = torch.autograd.grad(lower_order_grads, input_torchvar, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)
        higher_order_grads_array = np.array(higher_order_grads[0].data)


        temp_cov = copy.deepcopy(cov_data)
        temp_cov[dimension][t] = 0.0
        val += 0.5*np.sum(higher_order_grads_array*temp_cov[dimension])


    average_causal_effects.append(val)


average_causal_effects = np.array(average_causal_effects) - np.array(baseline_expectation_do_x)[:len(average_causal_effects)]
