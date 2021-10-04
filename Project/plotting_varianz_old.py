import numpy as np
import matplotlib.pyplot as plt


#--------41--------

varianz_ACE_41_0=[4.95002273847139e-09,1.1477397238671359e-08,8.258658642928586e-09,3.5243211059134208e-09,1.1018608785179207e-08]

varianz_ACE_41_1=[2.764800483654676e-08,9.044488167578828e-09,1.78063970629047e-08,7.032033769631351e-09,8.578897127509558e-09]

varianz_noACE_41=[7.61086661801085e-09,4.71393917382244e-09]

varianz_ACE_41=np.mean([varianz_ACE_41_0,varianz_ACE_41_1], axis=0)
var_varianz_ACE_41=np.var([varianz_ACE_41_0,varianz_ACE_41_1], axis=0)
varianz_noACE_41=np.mean(varianz_noACE_41)
var_varianz_noACE_41=np.var(varianz_noACE_41)
#--------369--------

varianz_ACE_369_0=[7.537361143999968e-05,0.00011676349952051792,0.0001545511045777673,6.231866284837027e-07,1.2068756609757187e-06]

varianz_noACE_369=[0.00010762616005825924]

varianz_ACE_369=np.mean([varianz_ACE_369_0], axis=0) #,varianz_ACE_369_1
var_varianz_ACE_369=np.var([varianz_ACE_369_0], axis=0) #,varianz_ACE_369_1
varianz_noACE_369=np.mean(varianz_noACE_369)
var_varianz_noACE_369=np.var(varianz_noACE_369)

#--------394--------

varianz_ACE_394_0=[1.1883812052305572e-08,3.5095474158768123e-06,2.184835350075369e-09,1.1558602321332662e-08,7.454082193848612e-09]

#varianz_noACE_394=[]

varianz_ACE_394=np.mean([varianz_ACE_394_0], axis=0)#,varianz_ACE_394_1
var_varianz_ACE_394=np.var([varianz_ACE_394_0], axis=0)
#varianz_noACE_394=np.mean(varianz_noACE_394)
#var_varianz_noACE_394=np.var(varianz_noACE_394)
#--------433--------

varianz_ACE_433_0=[2.361726381952218e-08,6.481950835819864e-08,2.4118114537193197e-08,8.406600530627942e-06,4.644143618396136e-09]

varianz_ACE_433_1=[1.4326055237863804e-07,8.783725498963947e-06,1.297671573257889e-06,9.345073884452524e-06,1.4026770367095202e-08]

varianz_noACE_433=[7.301197374487101e-06,8.26884709979183e-09]

varianz_ACE_433=np.mean([varianz_ACE_433_0,varianz_ACE_433_1], axis=0)
var_varianz_ACE_433=np.var([varianz_ACE_433_0,varianz_ACE_433_1], axis=0)
varianz_noACE_433=np.mean(varianz_noACE_433)
var_varianz_noACE_433=np.var(varianz_noACE_433)

#--------No Shift--------

varianz_ACE_0=[2.5217296117440853e-06,1.0902450364713216e-05,5.405325758810404e-06,1.5299942613063174e-06,1.4829219132661312e-06]

varianz_ACE_1=[3.2779818244177525e-05,7.371984254084081e-07,1.675057819504363e-05,2.554824252007473e-07,7.676848261216518e-06]

varianz_noACE=[5.191011699009859e-06,4.939382730946506e-06]

varianz_ACE=np.mean([varianz_ACE_0,varianz_ACE_1], axis=0)
var_varianz_ACE=np.var([varianz_ACE_0,varianz_ACE_1], axis=0)
varianz_noACE=np.mean(varianz_noACE)
var_varianz_noACE=np.var(varianz_noACE)



x_axis=[1,2,3,4,5]
my_xticks = ['Factor 0.1','Factor 1','Factor 10','Factor 100','Factor 1000']

#--------41--------

plt.xticks(x_axis, my_xticks)


varianz_noACE_41=[varianz_noACE_41,varianz_noACE_41,varianz_noACE_41,varianz_noACE_41,varianz_noACE_41]

plt.plot(x_axis, varianz_noACE_41, label="Without ACE")
plt.fill_between(x_axis, varianz_noACE_41-var_varianz_noACE_41, varianz_noACE_41+var_varianz_noACE_41,  color='grey')

plt.plot(x_axis, varianz_ACE_41, label="With ACE")
plt.fill_between(x_axis, varianz_ACE_41-var_varianz_ACE_41, varianz_ACE_41+var_varianz_ACE_41,  color='grey')


plt.xlabel('Factors')
plt.ylabel('Varianz')
plt.legend(loc='upper right')
plt.savefig('Varianz_41.png')
plt.close()

#--------369--------

plt.xticks(x_axis, my_xticks)


varianz_noACE_369=[varianz_noACE_369,varianz_noACE_369,varianz_noACE_369,varianz_noACE_369,varianz_noACE_369]

plt.plot(x_axis, varianz_noACE_369, label="Without ACE")
plt.fill_between(x_axis, varianz_noACE_369-var_varianz_noACE_369, varianz_noACE_369+var_varianz_noACE_369,  color='grey')

plt.plot(x_axis, varianz_ACE_369, label="With ACE")
plt.fill_between(x_axis, varianz_ACE_369-var_varianz_ACE_369, varianz_ACE_369+var_varianz_ACE_369,  color='grey')


plt.xlabel('Factors')
plt.ylabel('Varianz')
plt.legend(loc='upper right')
plt.savefig('Varianz_369.png')
plt.close()


#--------394--------

plt.xticks(x_axis, my_xticks)


#varianz_noACE_394=[varianz_noACE_394,varianz_noACE_394,varianz_noACE_394,varianz_noACE_394,varianz_noACE_394]

#plt.plot(x_axis, varianz_noACE_394, label="Without ACE")
#plt.fill_between(x_axis, varianz_noACE_394-var_varianz_noACE_394, varianz_noACE_394+var_varianz_noACE_394,  color='grey')

plt.plot(x_axis, varianz_ACE_394, label="With ACE")
plt.fill_between(x_axis, varianz_ACE_394-var_varianz_ACE_394, varianz_ACE_394+var_varianz_ACE_394,  color='grey')


plt.xlabel('Factors')
plt.ylabel('Varianz')
plt.legend(loc='upper right')
plt.savefig('Varianz_394.png')
plt.close()

#--------433--------

plt.xticks(x_axis, my_xticks)


varianz_noACE_433=[varianz_noACE_433,varianz_noACE_433,varianz_noACE_433,varianz_noACE_433,varianz_noACE_433]

plt.plot(x_axis, varianz_noACE_433, label="Without ACE")
plt.fill_between(x_axis, varianz_noACE_433-var_varianz_noACE_433, varianz_noACE_433+var_varianz_noACE_433,  color='grey')

plt.plot(x_axis, varianz_ACE_433, label="With ACE")
plt.fill_between(x_axis, varianz_ACE_433-var_varianz_ACE_433, varianz_ACE_433+var_varianz_ACE_433,  color='grey')


plt.xlabel('Factors')
plt.ylabel('Varianz')
plt.legend(loc='upper right')
plt.savefig('Varianz_433.png')
plt.close()

#--------No Shift--------

plt.xticks(x_axis, my_xticks)


varianz_noACE=[varianz_noACE,varianz_noACE,varianz_noACE,varianz_noACE,varianz_noACE]

plt.plot(x_axis, varianz_noACE, label="Without ACE")
plt.fill_between(x_axis, varianz_noACE-var_varianz_noACE, varianz_noACE+var_varianz_noACE,  color='grey')

plt.plot(x_axis, varianz_ACE, label="With ACE")
plt.fill_between(x_axis, varianz_ACE-var_varianz_ACE, varianz_ACE+var_varianz_ACE,  color='grey')


plt.xlabel('Factors')
plt.ylabel('Varianz')
plt.legend(loc='upper right')
plt.savefig('Varianz_noshift.png')
plt.close()
