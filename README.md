# Causal-Regularization-Master-Thesis

Paper: https://wiki.tum.de/display/ldv/Causal+Regularization+in+Deep+Learning+Using+the+Average+Causal+Effect

Code for Causal Inference in Deep Neural Nets 

Causal Interpretability aims to make decisions of algorithms interpretable by investigating what would have happened under different circumstances. These varying circumstances can be manipulations on the algorithm to assess its causality. In this thesis, I include the causal interpretability mechanism called the Average Causal Effect (ACE) into the training of a neural net. To assess the causality of the model, the ACE uses so-called interventions to manipulate the neural net. The goal is to determine more causal weights and biases during the model training. Using this approach, I evaluate whether including a causal interpretability mechanism as a regularization increases the overall causality of the model. Moreover, I investigate whether this improvement in causality also impacts the generalization ability of the neural net. The developed approach is compared to a standard neural net as well as L1 and L2 regularized neural nets. Furthermore, I conduct these experiments with well-balanced datasets, datasets with a prior probability shift, and datasets with a covariate shift. For all datasets, the results show that the presented causal regularization approach is able to improve the overall causality of the neural net. However, the distribution of the shifted training data highly affects the generalization ability. With an increasing variance of the distribution, the developed approach shows significantly lower test Mean Squared Errors than for training data with less variance. This is because the interventions applied by the ACE depend on the variance of the training data distribution.
