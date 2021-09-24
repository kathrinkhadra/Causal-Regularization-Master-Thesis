import numpy as np
import matplotlib.pyplot as plt
import re


#-----------------------factor1000---------------------------

ACEvalue_withACE_factor1000=[1.6415071595899459e-06, 1.7363781728116276e-06, 1.9570307900833207e-06, 2.017578090558428e-06, 1.9830747398456397e-06, 2.00841641303851e-06, 2.106973105300458e-06, 2.219927549941358e-06, 2.321629022103018e-06, 2.446405842766515e-06, 2.605174285678815e-06, 2.786051590950922e-06, 2.983826885427591e-06, 3.2107579080432847e-06, 3.4726920722473796e-06, 3.779507294444374e-06, 4.128494743785802e-06, 4.5275567171706645e-06, 4.986245434290774e-06, 5.510700369470207e-06, 6.106295292603082e-06, 6.784482340533349e-06, 7.5562301357183664e-06, 8.437217290234477e-06, 9.447151165594842e-06, 1.060471567447078e-05, 1.1935221470016251e-05, 1.3459124501083116e-05, 1.5206293533831047e-05, 1.720963381876876e-05, 1.9501831972287895e-05, 2.2124700926279454e-05, 2.5123765172709106e-05, 2.8547881032064777e-05, 3.2447857257276514e-05, 3.686367374438909e-05, 4.183525625983623e-05, 4.7393607589262064e-05, 5.353110616144842e-05, 6.021249559312368e-05, 6.736698399199858e-05, 7.487591903119134e-05, 8.258129575301122e-05, 9.029743187630937e-05, 9.784438087121801e-05]

#ACEvalue_withACE_factor1000_1=
#ACEvalue_withACE_factor1000_2

test_MSE_withACE_factor1000=[0.0300]

#-----------------------factor100---------------------------

ACEvalue_withACE_factor100=[1.435678426250828e-06, 1.4835255363815853e-06, 1.5839720367730272e-06, 1.6072166519399726e-06, 1.5614751167432373e-06, 1.544440814099787e-06, 1.567056850708723e-06, 1.6079488157903862e-06, 1.6541642173198357e-06, 1.7092424608513172e-06, 1.789503287389678e-06, 1.8893866495140118e-06, 2.0087296949308526e-06, 2.150378924612273e-06, 2.3185255493097106e-06, 2.5140757312008014e-06, 2.748813749836416e-06, 3.0380747135950566e-06, 3.3774683185409893e-06, 3.774039616153809e-06, 4.23643295083395e-06, 4.775012554894686e-06, 5.403446141810053e-06, 6.140413616357719e-06, 7.001324265285683e-06, 8.008193831180512e-06, 9.18858360325814e-06, 1.0574287143317232e-05, 1.2203794585915727e-05, 1.4117095837971342e-05, 1.6363467410467856e-05, 1.90031331516863e-05, 2.2103988789852004e-05, 2.5747183862985674e-05, 3.0017091194911635e-05, 3.50004157078919e-05, 4.077397827830078e-05, 4.7394279874182024e-05, 5.488684833920947e-05, 6.322078064046462e-05, 7.227994735538463e-05, 8.186303450468702e-05, 9.169058741220342e-05, 0.00010143241679254194, 0.00011075389558497702]

#ACEvalue_withACE_factor100_1=
#ACEvalue_withACE_factor100_2=

test_MSE_withACE_factor100=[0.0291]


#-----------------------factor10---------------------------


ACEvalue_withACE_factor10=[1.5936612806549547e-06, 1.5743156768730223e-06, 1.6048952392060482e-06, 1.5841603066961529e-06, 1.6142655227100434e-06, 1.6991395125874745e-06, 1.7940682411544397e-06, 1.9038338876561309e-06, 2.069109254157317e-06, 2.2671016123043624e-06, 2.499986588604595e-06, 2.781991979740087e-06, 3.1126019761050643e-06, 3.5047940291374596e-06, 3.969259517454594e-06, 4.51808420206529e-06, 5.171334364256372e-06, 5.950075405629056e-06, 6.883747062048937e-06, 8.005550184406406e-06, 9.36344325011069e-06, 1.0998763170444355e-05, 1.2977469697825584e-05, 1.5377977234665946e-05, 1.8290654473468475e-05, 2.1826110987990016e-05, 2.6113476112938082e-05, 3.129154997291547e-05, 3.749371189887665e-05, 4.48305219831105e-05, 5.333826212337237e-05, 6.2928067449402e-05, 7.333564579929421e-05, 8.408296708398533e-05, 9.453403048185966e-05, 0.00010402460833028329, 0.0001120582266534364, 0.0001184433546377476, 0.00012331711078804013, 0.00012703027958109673, 0.0001299714198966344, 0.00013246042055455688, 0.00013470099966118206, 0.00013682199382571142, 0.00013887615456314198]

#ACEvalue_withACE_factor10_1=
#ACEvalue_withACE_factor10_2=

test_MSE_withACE_factor10=[0.0276]#0.2179

#-----------------------factor1---------------------------
ACEvalue_withACE_factor1=[1.641945774536604e-06, 1.6212548951127441e-06, 1.5820622545413482e-06, 1.5839232782189732e-06, 1.620823216482761e-06, 1.6734751394576702e-06, 1.8007217752367634e-06, 1.972440072911139e-06, 2.1993261256690528e-06, 2.4889329416655176e-06, 2.8533143813025253e-06, 3.316572854292119e-06, 3.900772370777746e-06, 4.6539867861705145e-06, 5.6263333715013426e-06, 6.874580163617679e-06, 8.486707962093525e-06, 1.0587253514767168e-05, 1.3335922620742533e-05, 1.6953011894825804e-05, 2.1742059731496353e-05, 2.808920072571779e-05, 3.6409407071264436e-05, 4.7064836052041e-05, 6.022200991893627e-05, 7.552622164319518e-05, 9.18613447769035e-05, 0.00010731501846213568, 0.00011984019286108042, 0.00012840185794736985, 0.00013345391822564378, 0.00013636049436504286, 0.0001383740056184437, 0.00014024255622867304, 0.0001422437600707647, 0.00014437232933892032, 0.00014649875355211546, 0.0001485272097731592, 0.00015040207930694851, 0.0001520905694722551, 0.00015359817420003804, 0.00015495579494272633, 0.00015619084383732888, 0.0001572988416818398, 0.0001582660887687636]

#ACEvalue_withACE_factor1_1=
#ACEvalue_withACE_factor1_2=

test_MSE_withACE_factor1=[0.0233]

#-----------------------factor01---------------------------
ACEvalue_withACE_factor01=[1.483485274293712e-06, 1.4840990477729361e-06, 1.6943983621684196e-06, 1.7559998531507151e-06, 1.7043832340106435e-06, 1.716595892006646e-06, 1.8052506610620707e-06, 1.9164516866615544e-06, 2.0254286908125517e-06, 2.1460546319473237e-06, 2.308323775675446e-06, 2.5066365399046997e-06, 2.73140576707977e-06, 2.9900772438647238e-06, 3.2910791439696227e-06, 3.6398703748168066e-06, 4.045159362762935e-06, 4.51419495813525e-06, 5.080467071307065e-06, 5.736882151019216e-06, 6.502534536105039e-06, 7.397912925001911e-06, 8.445242442433747e-06, 9.66796101544313e-06, 1.1099286130991427e-05, 1.2780147146845494e-05, 1.4752471615142071e-05, 1.7066702888088484e-05, 1.978246495083319e-05, 2.2963159342761466e-05, 2.667807326550124e-05, 3.100153364167126e-05, 3.6006163188339476e-05, 4.17543054879018e-05, 4.828579182363902e-05, 5.559747102543548e-05, 6.364232472647879e-05, 7.231568855130897e-05, 8.14094238014975e-05, 9.066173794774721e-05, 9.977604887193367e-05, 0.00010847575641225197, 0.00011650378017458919, 0.00012369275720145955, 0.0001300460223051572]

#ACEvalue_withACE_factor01_1=
#ACEvalue_withACE_factor01_2=

test_MSE_withACE_factor01=[0.0289]

#-----------------------NoACE---------------------------

ACEvalue_noACE=[1.895351826945411e-06, 2.0768482039242025e-06, 2.1965122035466136e-06, 2.1707709955239013e-06, 2.2123552205956365e-06, 2.3528047076708043e-06, 2.487115185266541e-06, 2.6178440705646277e-06, 2.8041387515032096e-06, 3.0238707659437164e-06, 3.2817363812249038e-06, 3.6129008762830606e-06, 4.008401607889926e-06, 4.471447539343766e-06, 5.019080091108188e-06, 5.6606700438799806e-06, 6.4173666693173204e-06, 7.310147978342221e-06, 8.363431101699999e-06, 9.608777466619796e-06, 1.1082999482989616e-05, 1.2829120302404515e-05, 1.4896405700660887e-05, 1.7343117448640663e-05, 2.024018817555574e-05, 2.367040810710695e-05, 2.7742133837831408e-05, 3.256146553757387e-05, 3.821383375928361e-05, 4.476633143718255e-05, 5.225424610714915e-05, 6.061400014291155e-05, 6.967532497372203e-05, 7.913084765756909e-05, 8.8548910310116e-05, 9.744860396221172e-05, 0.00010539417126368937, 0.00011210474380505373, 0.0001175311838100414, 0.00012179400889309707, 0.0001251274310695919, 0.00012781925524049887, 0.000130094641451269, 0.00013211783723280675, 0.00013400067578638335]

#ACEvalue_noACE_1=

#ACEvalue_noACE_2=

without_ACE_test_MSE=[0.0277]

#-----------------------plotting---------------------------

#x_axis=np.arange(1, 451)
#plt.plot(x_axis,training_MSE_noACE_factor1000, linestyle='dashed',label="Without ACE")
#plt.plot(x_axis,training_MSE_withACE_factor1000, label="Factor 1000")
#plt.plot(x_axis,training_MSE_withACE_factor100, label="Factor 100")
#plt.plot(x_axis,training_MSE_withACE_factor10, label="Factor 10")
#plt.xlabel('Training epoch')
#plt.ylabel('Mean Square Error (MSE)')
#plt.legend(loc='upper right')
#plt.savefig('Training_entwicklung_layer.png')
#plt.close()


#ACEvalue_withACE_factor100=ACEvalue_withACE_factor100[0::10]
#ACEvalue_withACE_factor10=ACEvalue_withACE_factor10[0::10]
#ACEvalue_withACE_factor01=ACEvalue_withACE_factor01[0::10]

#ACEvalue_noACE_factor100=ACEvalue_noACE_factor100[0::10]
#ACEvalue_noACE_factor10=ACEvalue_noACE_factor10[0::10]
#ACEvalue_noACE_factor01=ACEvalue_noACE_factor01[0::10]
#print(len(ACEvalue_noACE_factor01))

ACEvalue_noACE=[ACEvalue_noACE]#,ACEvalue_noACE_1,ACEvalue_noACE_2]#,ACEvalue_noACE_factor10

ACEvalue_withACE_factor01=[ACEvalue_withACE_factor01]#,ACEvalue_withACE_factor01_1,ACEvalue_withACE_factor01_2]
ACEvalue_withACE_factor1=[ACEvalue_withACE_factor1]#,ACEvalue_withACE_factor1_1,ACEvalue_withACE_factor1_2]
ACEvalue_withACE_factor10=[ACEvalue_withACE_factor10]#,ACEvalue_withACE_factor10_1,ACEvalue_withACE_factor10_2]
ACEvalue_withACE_factor100=[ACEvalue_withACE_factor100]#,ACEvalue_withACE_factor100_1,ACEvalue_withACE_factor100_2]
ACEvalue_withACE_factor1000=[ACEvalue_withACE_factor1000]#,ACEvalue_withACE_factor1000_1,ACEvalue_withACE_factor1000_2]

var_noACE=np.var(ACEvalue_noACE, axis=0)
mean_noACE=np.mean(ACEvalue_noACE, axis=0)
#print(var.shape)
#print(mean.shape)
x_axis=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450]
#x_axis=np.arange(1, 451)
#plt.plot(x_axis,ACEvalue_withACE_factor1000, label="Factor 1000")
#plt.plot(x_axis,ACEvalue_withACE_factor100, label="Factor 100")
#plt.plot(x_axis,ACEvalue_withACE_factor10, label="Factor 10")
#plt.plot(x_axis,ACEvalue_withACE_factor1, label="Factor 1")
#plt.plot(x_axis,ACEvalue_withACE_factor01, label="Factor 0.1")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor10, axis=0), yerr=np.var(ACEvalue_withACE_factor10, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 10")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1, axis=0), yerr=np.var(ACEvalue_withACE_factor1, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor01, axis=0), yerr=np.var(ACEvalue_withACE_factor01, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 0.1")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor100, axis=0), yerr=np.var(ACEvalue_withACE_factor100, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 100")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1000, axis=0), yerr=np.var(ACEvalue_withACE_factor1000, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1000")
#plt.errorbar(x_axis, mean_noACE, yerr=var_noACE, linestyle='--', marker='s', markersize=2, label="Without ACE")
#plt.plot(x_axis,ACEvalue_noACE_factor1000, label="No ACE")

plt.plot(x_axis, np.mean(ACEvalue_withACE_factor10, axis=0), label="Factor 10")
plt.fill_between(x_axis, np.mean(ACEvalue_withACE_factor10, axis=0)-np.var(ACEvalue_withACE_factor10, axis=0), np.mean(ACEvalue_withACE_factor10, axis=0)+np.var(ACEvalue_withACE_factor10, axis=0), edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x_axis, np.mean(ACEvalue_withACE_factor1, axis=0), label="Factor 1")
plt.fill_between(x_axis, np.mean(ACEvalue_withACE_factor1, axis=0)-np.var(ACEvalue_withACE_factor1, axis=0), np.mean(ACEvalue_withACE_factor1, axis=0)+np.var(ACEvalue_withACE_factor1, axis=0), edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x_axis, np.mean(ACEvalue_withACE_factor01, axis=0), label="Factor 0.1")
plt.fill_between(x_axis, np.mean(ACEvalue_withACE_factor01, axis=0)-np.var(ACEvalue_withACE_factor01, axis=0), np.mean(ACEvalue_withACE_factor01, axis=0)+np.var(ACEvalue_withACE_factor01, axis=0), edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x_axis, np.mean(ACEvalue_withACE_factor100, axis=0), label="Factor 100")
plt.fill_between(x_axis, np.mean(ACEvalue_withACE_factor100, axis=0)-np.var(ACEvalue_withACE_factor100, axis=0), np.mean(ACEvalue_withACE_factor100, axis=0)+np.var(ACEvalue_withACE_factor100, axis=0), edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x_axis, np.mean(ACEvalue_withACE_factor1000, axis=0), label="Factor 1000")
plt.fill_between(x_axis, np.mean(ACEvalue_withACE_factor1000, axis=0)-np.var(ACEvalue_withACE_factor1000, axis=0), np.mean(ACEvalue_withACE_factor1000, axis=0)+np.var(ACEvalue_withACE_factor1000, axis=0), edgecolor='#CC4F1B', facecolor='#FF9848')


plt.plot(x_axis, mean_noACE, label="Without ACE")
plt.fill_between(x_axis, mean_noACE-var_noACE, mean_noACE+var_noACE, edgecolor='#CC4F1B', facecolor='#FF9848')


plt.xlabel('Training epoch')
plt.ylabel('Average causal effect (ACE)')
plt.legend(loc='upper left')
plt.savefig('ACE_entwicklung_layer_noshift.png')
plt.close()

#x_axis=[1,2,3,4]
#my_xticks = ['Factor 1','Factor 1000','Factor 1000','Factor 10000']
#with_ACE_test_MSE=[test_MSE_withACE_factor1,test_MSE_withACE_factor1000,test_MSE_withACE_factor1000_1,test_MSE_withACE_factor10000]
#without_ACE_test_MSE=[test_MSE_noACE_factor1,test_MSE_noACE_factor1000,test_MSE_noACE_factor1000_1,test_MSE_noACE_factor10000]

x_axis=[1,2,3,4,5]
my_xticks = ['Factor 0.1','Factor 1','Factor 10','Factor 100','Factor 1000']
means_with_ACE_test_MSE=[np.mean(test_MSE_withACE_factor01),np.mean(test_MSE_withACE_factor1),np.mean(test_MSE_withACE_factor10),np.mean(test_MSE_withACE_factor100),np.mean(test_MSE_withACE_factor1000)]
var_with_ACE_test_MSE=[np.var(test_MSE_withACE_factor01),np.var(test_MSE_withACE_factor1),np.var(test_MSE_withACE_factor10),np.var(test_MSE_withACE_factor100),np.var(test_MSE_withACE_factor1000)]
#without_ACE_test_MSE=[test_MSE_noACE_factor01,test_MSE_noACE_factor1,[test_MSE_noACE_factor10,test_MSE_noACE_factor10],[test_MSE_noACE_factor100,test_MSE_noACE_factor100],[test_MSE_noACE_factor1000,test_MSE_noACE_factor1000]]

#print(without_ACE_test_MSE)
#without_ACE_test_MSE = [x for sublist in without_ACE_test_MSE for x in sublist]
#without_ACE_test_MSE=np.array(without_ACE_test_MSE).flatten()
#print(without_ACE_test_MSE)
#print(without_ACE_test_MSE.ravel())

var=np.var(without_ACE_test_MSE)
mean=np.mean(without_ACE_test_MSE)
without_ACE_test_MSE=[mean,mean,mean,mean,mean]

plt.xticks(x_axis, my_xticks)
#plt.plot(x_axis,with_ACE_test_MSE, marker='o', markersize=2,label="With ACE")
#plt.plot(x_axis,without_ACE_test_MSE, marker='o', label="Without ACE")
#plt.errorbar(x_axis, means_with_ACE_test_MSE, yerr=var_with_ACE_test_MSE, linestyle='-', marker='s', markersize=2, label="With ACE")
#plt.errorbar(x_axis, without_ACE_test_MSE, yerr=var, linestyle='--', marker='s', markersize=2, label="Without ACE")
plt.plot(x_axis, without_ACE_test_MSE, label="Without ACE")
plt.fill_between(x_axis, without_ACE_test_MSE-var, without_ACE_test_MSE+var, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x_axis, means_with_ACE_test_MSE, label="With ACE")



plt.xlabel('Factor')
plt.ylabel('Mean Square Error (MSE)')
plt.legend(loc='upper right')
plt.savefig('Test_entwicklung_layer_noshift.png')
plt.close()
