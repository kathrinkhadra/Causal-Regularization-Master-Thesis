import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from arangopipe.arangopipe_analytics.rf_dataset_shift_detector import RF_DatasetShiftDetector
import pandas as pd
import scipy as sp
import sklearn

import NeuralNet
import datapreprocessing
import causal

test_size=0.9
get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)
get_data.splitting_data_noshift()
#get_data.target_shift(50)
#get_data.target_shift_big(50)
#get_data.target_shift_mid(50,100)
split=0
#get_data.dataset_shift(split)#41,243,394,433,369
#get_data.dataset_shift(369)
#get_data.dataset_shift(433)

a, inputs_test, b, target_test = train_test_split(get_data.inputs_test, get_data.target_test , test_size=50, random_state=0)

feature_names=["CRIM","ZN","INDUS","CHAS","TAX","PTRATIO","B","LSTAT","NOX","RM","AGE","DIS","RAD"]

splits=[369,433,394,0]
mutual_info_array=[]
rfd = RF_DatasetShiftDetector()
#scaler = MinMaxScaler(feature_range=(0, 1))

splits=[0,369,433,394]
for i in range(13):#[9,2,12,7,4]:
    fig, axs = plt.subplots(2, 2)
    for ax in axs:
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_xlabel('Values of Samples', fontsize=6)
            a.set_ylabel('Density', fontsize=6)
    for ind,split in enumerate(splits):
        if split==0:
            get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)
            get_data.splitting_data_noshift()
            g=0
            c=0
        else:
            get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)
            get_data.dataset_shift(split)
        if split==369:
            g=0
            c=1
        if split==433:
            g=1
            c=0
        if split==394:
            g=1
            c=1
        axs[g,c].hist([get_data.inputs_test[:,i],get_data.inputs_training[:,i]], label=["Test data","Training data"])
        if split==0:
            title="Well Balanced"
        else:
            title="Split" + str(split)
        axs[g, c].set_title(title, fontsize=6)#axs[0, 0].set_title("\n".join(wrap(title, 60)))

    name='hist/histogram_'+str(i)+'_.pgf'
    title='Histogram of Feature '+str(feature_names[i])
    fig.tight_layout()
    handles, labels = axs[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", borderaxespad=0.1)#, bbox_to_anchor=(1.04,1)) #,loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure
    fig.subplots_adjust(right=0.85)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.88)#0.88
    fig.savefig(name, dpi=300)
    plt.close()

'''
for split in splits:
    mutual_info=[]
    if split==0:
        get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)
        get_data.splitting_data_noshift()
    else:
        get_data= datapreprocessing.Dataprep(0,0,0,0,0,0,test_size)
        get_data.dataset_shift(split)
    for i in [9,2,12,7,4]:#range(13):
        #print(inputs_test[:,i].shape)
        #print(get_data.inputs_training[:,i].shape)
    #    print(i)
        #print(normalized_mutual_info_score(inputs_test[:,i],get_data.inputs_training[:,i]))
        #mutual_info.append(mutual_info_score(inputs_test[:,i],get_data.inputs_training[:,i]))
        mutual_info.append(rand_score(inputs_test[:,i],get_data.inputs_training[:,i]))#adjusted_
        #df1=pd.DataFrame(get_data.inputs_test[:,i])
        #df1=df1.query
        #mutual_info.append(rfd.detect_dataset_shift(, pd.DataFrame(get_data.inputs_training[:,i])))
        #plt.hist([get_data.inputs_test[:,i],get_data.inputs_training[:,i]], label=["Test data","Training data"])#density=True
        #plt.hist(get_data.inputs_training[:,i], label="Training", density=True, alpha=0.5)
        #print(inputs_test[:,i])
        #print(get_data.inputs_training[:,i])
    #mutual_info[8]=-mutual_info[8]

    mutual_info_array.append(mutual_info)#(mutual_info  - min(mutual_info)) / (max(mutual_info) - min(mutual_info))

    #my_xticks = ['e-05',' e-04',' e-03',' e-02',' e-01',' 1',' 10',' 100']
    #plt.xticks(i, my_xticks)
    #i=range(13)
    #plt.bar(i,mutual_info)
'''
'''
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    #plt.xlabel('Feature')
    #plt.ylabel('Mutual Information')
    title="Histogram of Feature "+str(feature_names[i])+" for Split "+str(split)
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.04,1))
    name='hist/feature_mutual_info_'+str(split)+'_'+str(i)+'.png'
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()
'''
'''
i=range(13)
my_xticks = ["CRIM","ZN","INDUS","CHAS","TAX","PTRATIO","B","LSTAT","NOX","RM","AGE","DIS","RAD","TARGET"]


fig, axs = plt.subplots(2, 2)
fig.tight_layout()

for ax in axs:
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xticks(i)
        a.set_xticklabels(my_xticks)
        a.set_xlabel('Feature', fontsize=6)
        a.set_ylabel('Mutual Information', fontsize=6)
        a.tick_params(axis='x', which='major', labelsize=4)
        a.axhline(min(mutual_info_array[3]),linestyle='--',color="orange")
axs[0, 0].bar(i,mutual_info_array[3])
#axs[0, 0].set_xticks(i)
#axs[0, 0].set_xticklabels(my_xticks)
#axs[0, 0].set_xlabel('Feature', fontsize=6)
#axs[0, 0].set_ylabel('Mutual Information', fontsize=6)
title="Mutual Information of Feature when well Balanced"
axs[0, 0].set_title(title, fontsize=6)#axs[0, 0].set_title("\n".join(wrap(title, 60)))
#axs[0, 0].tick_params(axis='x', which='major', labelsize=4)
#plt.legend(loc='upper left', bbox_to_anchor=(1.04,1))

axs[0, 1].bar(i,mutual_info_array[2])
axs[0, 1].set_xticks(i)
axs[0, 1].set_xticklabels(my_xticks)
axs[0, 1].set_xlabel('Feature', fontsize=6)
axs[0, 1].set_ylabel('Mutual Information', fontsize=6)
title="Mutual Information of Feature with Split 1"
axs[0, 1].set_title(title, fontsize=6)
axs[0, 1].tick_params(axis='x', which='major', labelsize=4)

axs[1, 0].bar(i,mutual_info_array[1])
axs[1, 0].set_xticks(i)
axs[1, 0].set_xticklabels(my_xticks)
axs[1, 0].set_xlabel('Feature', fontsize=6)
axs[1, 0].set_ylabel('Mutual Information', fontsize=6)
title="Mutual Information of Feature with Split 2"
axs[1, 0].set_title(title, fontsize=6)
axs[1, 0].tick_params(axis='x', which='major', labelsize=4)

axs[1, 1].bar(i,mutual_info_array[0])
axs[1, 1].set_xticks(i)
axs[1, 1].set_xticklabels(my_xticks)
axs[1, 1].set_xlabel('Feature', fontsize=6)
axs[1, 1].set_ylabel('Mutual Information', fontsize=6)
title="Mutual Information of Feature with Split 3"
axs[1, 1].set_title(title, fontsize=6)
axs[1, 1].tick_params(axis='x', which='major', labelsize=4)



name='hist/feature_mutual_info.png'
fig.savefig(name, dpi=300)
plt.close()
'''

#feature_order=[11,0,9,5,10,1,2,8,12,7,6,3,4]
i=range(13)
my_xticks = ["CRIM","ZN","INDUS","CHAS","TAX","PTRATIO","B","LSTAT","NOX","RM","AGE","DIS","RAD","TARGET"]
my_xticks = [my_xticks[11],my_xticks[0],my_xticks[9],my_xticks[5],my_xticks[10],my_xticks[1],my_xticks[2],my_xticks[8],my_xticks[12],my_xticks[7],my_xticks[6],my_xticks[3],my_xticks[4]]

MSE=[0.0194,0.0253,0.0267,0.0272,0.0278,0.0282,0.0284,0.0289,0.0293,0.0295,0.0304,0.0304,0.0319]

plt.scatter(i,MSE)
plt.xticks(i, my_xticks)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('Feature')
plt.ylabel('Mean Squared Error (MSE)')
title="MSE for dropping each feature"
plt.title(title)
plt.legend(loc='upper left', bbox_to_anchor=(1.04,1))
name='hist/feature_importance.pgf'
plt.savefig(name, dpi=300, bbox_inches="tight")
plt.close()
