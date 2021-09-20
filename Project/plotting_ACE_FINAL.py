import numpy as np
import matplotlib.pyplot as plt
import re


#--------------Factor01-----------------

#--------------369-----------------

test_MSE_withACE_factor01_369=[]

#ACEvalue_withACE_factor01_369_0=

#ACEvalue_withACE_factor01_369_0=ACEvalue_withACE_factor01_369_0[0::10]

#ACEvalue_withACE_factor01_369_1=
#ACEvalue_withACE_factor01_369_2=

#ACEvalue_withACE_factor01_369=[ACEvalue_withACE_factor01_369_0,ACEvalue_withACE_factor01_369_1,ACEvalue_withACE_factor01_369_2]
#--------------433-----------------

test_MSE_withACE_factor01_433=[]

#ACEvalue_withACE_factor01_433_0=
#ACEvalue_withACE_factor01_433_1=
#ACEvalue_withACE_factor01_433_2=

#ACEvalue_withACE_factor01_433=[ACEvalue_withACE_factor01_433_0,ACEvalue_withACE_factor01_433_1,ACEvalue_withACE_factor01_433_2]

#--------------41-----------------

test_MSE_withACE_factor01_41=[]

#ACEvalue_withACE_factor01_41_0=[9.58528772950455e-06, -1.0981058083143886e-06, -1.561972910980302e-05, -3.12566474978408e-05, -6.0182372191941107e-05, -9.162593153100906e-05, -8.357148614030309e-05, -6.3299415702444e-05, -5.870104327712505e-05, -6.723599864784503e-05, -5.9592590131586726e-05, -6.135034352088693e-05, -7.205456913977449e-05, -7.669116483198506e-05, -7.720129424562423e-05, -9.838080697231412e-05, -8.512573487788213e-05, -7.819847410785481e-05, -7.315671046382433e-05, -7.597281914653183e-05, -7.6596359590076e-05, -6.443185407923551e-05, -4.909855680291523e-05, -3.6401653352740785e-05, -4.177104872611474e-05, -5.277029815319173e-05, -5.28382225285238e-05, -5.8316695169505255e-05, -4.505672992496348e-05, -4.2162393719248516e-05, -4.688996637444709e-05, -4.058582971473901e-05, -3.6804331443800115e-05, -3.723995701747395e-05, -4.293327320528008e-05, -4.741162978676037e-05, -5.614367109243256e-05, -6.03836388650694e-05, -5.681515529947942e-05, -5.051395403186141e-05, -5.6749321818300146e-05, -5.8651768308195044e-05, -5.7978436255676345e-05, -5.965935331457477e-05, -5.821533979510555e-05]

#ACEvalue_withACE_factor01_41_1=[-1.724521133633482e-05, -1.5135526413611034e-05, -6.04207483993438e-06, -3.1131867998841916e-06, -5.805244199495096e-06, -1.0185038048944339e-05, -1.2826024146073587e-05, -3.214555393822222e-05, -3.861120648513176e-05, -4.8724647149455426e-05, -4.5687822778069896e-05, -4.5320051508616464e-05, -6.820118133300195e-05, -3.884232670489925e-05, -3.2930453261290397e-05, -2.9951867599663104e-05, -4.778963698434968e-05, -4.734147236709254e-05, -5.8731387236981315e-05, -6.993804338749934e-05, -8.608796259752173e-05, -0.00011699620941617807, -0.0001356366751015743, -0.0001304632462789755, -0.00011469226621980555, -0.00010828825338457136, -0.0001039639834398221, -0.00010002841655161862, -9.829977987765572e-05, -0.00010347898192013541, -0.00011156444372281563, -0.00011492470694817796, -0.00012673298676787935, -0.00012681471077268588, -0.00012792059091550152, -0.00012825638218783098, -0.0001259054288526861, -0.0001257058156933193, -0.00012448442242796672, -0.00012167961604525458, -0.00012301865811209484, -0.00012582584903211925, -0.0001302990576126284, -0.00013470940418533524, -0.0001334224264960547]

#ACEvalue_withACE_factor01_41_2=[-1.4215607322767967e-05, -1.5960292459847206e-05, -8.013008253444145e-06, -1.3623480418276206e-05, -1.3699107700389853e-05, -1.6132540387309574e-05, -2.508671724353367e-05, -3.421297532655108e-05, -3.150754336342637e-05, -3.4660468279085176e-05, -2.7568515102271073e-05, -3.3557419119336963e-06, 2.4544126152771694e-06, -9.205101333406065e-08, -8.230694119283485e-06, -1.0987577200598212e-05, 7.237736779909696e-06, 3.7415298387167175e-05, 5.3938358008574125e-05, 5.820227287004792e-05, 3.631522358601875e-05, 3.1633092082258485e-05, 1.631151349919238e-05, 6.7629055251983934e-06, -2.3980918291351537e-06, -4.6314429170890104e-06, -1.971285697048034e-05, -2.6547153858257645e-05, -2.3720227910570204e-05, -3.323632770726078e-05, -4.198845051224199e-05, -4.645587620217891e-05, -5.335000576584086e-05, -5.8548867364175216e-05, -6.396929889574103e-05, -6.180610617825469e-05, -5.2994288772063385e-05, -4.5931678870314914e-05, -4.301490939181218e-05, -3.982122792177105e-05, -3.562079518416906e-05, -2.9167823690656372e-05, -2.0753748240086595e-05, -1.5194844953947902e-05, -3.564942575385876e-06]

#ACEvalue_withACE_factor01_41=[ACEvalue_withACE_factor01_41_0,ACEvalue_withACE_factor01_41_1,ACEvalue_withACE_factor01_41_2]

#--------------Factor1-----------------

#--------------369-----------------

test_MSE_withACE_factor1_369=[]

#ACEvalue_withACE_factor1_369_0=
#ACEvalue_withACE_factor1_369_1=

#ACEvalue_withACE_factor1_369_2=
#ACEvalue_withACE_factor1_369=[ACEvalue_withACE_factor1_369_0,ACEvalue_withACE_factor1_369_1]
#--------------433-----------------

test_MSE_withACE_factor1_433=[]

#ACEvalue_withACE_factor1_433_0=
#ACEvalue_withACE_factor1_433_1=
#ACEvalue_withACE_factor1_433_2=
#ACEvalue_withACE_factor1_433=[ACEvalue_withACE_factor1_433_0,ACEvalue_withACE_factor1_433_1,ACEvalue_withACE_factor1_433_2]

#--------------41-----------------

test_MSE_withACE_factor1_41=[]

#ACEvalue_withACE_factor1_41_0=
#ACEvalue_withACE_factor1_41_1=
#ACEvalue_withACE_factor1_41_2=

#ACEvalue_withACE_factor1_41=[ACEvalue_withACE_factor1_41_0,ACEvalue_withACE_factor1_41_1,ACEvalue_withACE_factor1_41_2]

#--------------Factor10-----------------

#--------------369-----------------

test_MSE_withACE_factor10_369=[]

#ACEvalue_withACE_factor10_369_0=
#ACEvalue_withACE_factor10_369_0=ACEvalue_withACE_factor10_369_0[0::10]

#ACEvalue_withACE_factor10_369_1=

#ACEvalue_withACE_factor10_369_2=

#ACEvalue_withACE_factor10_369=[ACEvalue_withACE_factor10_369_0,ACEvalue_withACE_factor10_369_1,ACEvalue_withACE_factor10_369_2]
#--------------433-----------------

test_MSE_withACE_factor10_433=[]

#ACEvalue_withACE_factor10_433_0=
#ACEvalue_withACE_factor10_433_1=
#ACEvalue_withACE_factor10_433_2=
#ACEvalue_withACE_factor10_433=[ACEvalue_withACE_factor10_433_0,ACEvalue_withACE_factor10_433_1,ACEvalue_withACE_factor10_433_2]
#--------------41-----------------

test_MSE_withACE_factor10_41=[]

#ACEvalue_withACE_factor10_41_0=
#ACEvalue_withACE_factor10_41_1=
#ACEvalue_withACE_factor10_41_2=

#ACEvalue_withACE_factor10_41=[ACEvalue_withACE_factor10_41_0,ACEvalue_withACE_factor10_41_1,ACEvalue_withACE_factor10_41_2]

#--------------Factor100-----------------

#--------------369-----------------

test_MSE_withACE_factor100_369=[]

#ACEvalue_withACE_factor100_369_0=
#ACEvalue_withACE_factor100_369_0=ACEvalue_withACE_factor100_369_0[0::10]

#ACEvalue_withACE_factor100_369_1=
#ACEvalue_withACE_factor100_369_2=
#ACEvalue_withACE_factor100_369=[ACEvalue_withACE_factor100_369_0,ACEvalue_withACE_factor100_369_1,ACEvalue_withACE_factor100_369_2]

#--------------433-----------------

test_MSE_withACE_factor100_433=[]

#ACEvalue_withACE_factor100_433_0=
#ACEvalue_withACE_factor100_433_1=
#ACEvalue_withACE_factor100_433_2=
#ACEvalue_withACE_factor100_433=[ACEvalue_withACE_factor100_433_0,ACEvalue_withACE_factor100_433_1,ACEvalue_withACE_factor100_433_2]

#--------------41-----------------

test_MSE_withACE_factor100_41=[]

#ACEvalue_withACE_factor100_41_0=
#ACEvalue_withACE_factor100_41_1=
#ACEvalue_withACE_factor100_41_2=
#ACEvalue_withACE_factor100_41=[ACEvalue_withACE_factor100_41_0,ACEvalue_withACE_factor100_41_1,ACEvalue_withACE_factor100_41_2]

#--------------Factor1000-----------------

#--------------369-----------------
test_MSE_withACE_factor1000_369=[]

#ACEvalue_withACE_factor1000_369_0=
#ACEvalue_withACE_factor1000_369_1=
#ACEvalue_withACE_factor1000_369_2=
#ACEvalue_withACE_factor1000_369=[ACEvalue_withACE_factor1000_369_1,ACEvalue_withACE_factor1000_369_2]
#--------------433-----------------

test_MSE_withACE_factor1000_433=[]

#ACEvalue_withACE_factor1000_433_0=
#ACEvalue_withACE_factor1000_433_1=

#ACEvalue_withACE_factor1000_433_2=

#ACEvalue_withACE_factor1000_433=[ACEvalue_withACE_factor1000_433_0,ACEvalue_withACE_factor1000_433_1,ACEvalue_withACE_factor1000_433_2]

#--------------41-----------------

test_MSE_withACE_factor1000_41=[]

#ACEvalue_withACE_factor1000_41_0=
#ACEvalue_withACE_factor1000_41_1=
#ACEvalue_withACE_factor1000_41_2=

#ACEvalue_withACE_factor1000_41=[ACEvalue_withACE_factor1000_41_0,ACEvalue_withACE_factor1000_41_1,ACEvalue_withACE_factor1000_41_2]

#--------------NoACE-----------------

#--------------369-----------------

test_MSE_noACE_369=[0.1203,0.0528,0.1146,0.0730,0.0855,0.1508,0.0692]

ACEvalue_noACE_369_0=[8.844475905919152e-07, 5.370918256917558e-06, 7.630458802213138e-06, 1.171494228157423e-05, 1.086357798626814e-05, 1.2741113532356861e-05, 1.0187949301175572e-05, 1.3417551107073268e-05, 9.360375374939845e-06, 1.2035256003786796e-05, 1.3078929435609489e-05, 1.1748822278859245e-05, 1.067916498324473e-05, 1.038963031288632e-05, 1.067230327447515e-05, 9.660418542356044e-06, 9.700171730750453e-06, 9.437238757292482e-06, 1.165548979430236e-05, 1.7836081147232593e-05, 2.0362180983454456e-05, 1.735850536539499e-05, 1.6363289157506498e-05, 1.531058856018923e-05, 1.3932666712880669e-05, 1.312774801201224e-05, 1.3624527536927709e-05, 1.474506579563018e-05, 1.5004461105786179e-05, 1.5188811308083277e-05, 1.5940960746420428e-05, 1.8856918531118015e-05, 2.2112932410214757e-05, 2.513209432643844e-05, 2.6661180674081857e-05, 2.3703556356429547e-05, 1.9551097565219683e-05, 1.5307882418503818e-05, 1.138665365124284e-05, 9.074250777894038e-06, 5.9676479034273544e-06, 4.407471463083452e-06, 5.546438404445819e-06, 5.7699463173374975e-06, 6.364627313650847e-06, 7.384658792805209e-06, 8.141484606987995e-06, 9.391608384636057e-06, 1.179942725436986e-05, 1.3892426225074952e-05, 1.6720873446466822e-05, 2.1083564327572663e-05, 2.3573039690309852e-05, 2.5008708467123067e-05, 2.7640599592682304e-05, 2.792454872608275e-05, 2.707157207170429e-05, 2.8248421115078207e-05, 2.994129825739095e-05, 3.08401261700821e-05, 3.077379402324255e-05, 3.109078161734973e-05, 3.17354698541709e-05, 3.27643332179561e-05, 3.371329148558465e-05, 3.460179216410789e-05, 3.526211642290657e-05, 3.563051231163434e-05, 3.684664968445895e-05, 3.7717544587357704e-05, 3.849537656813285e-05, 3.901468973472268e-05, 3.938987491083073e-05, 4.081799742562819e-05, 4.1870408498578595e-05, 4.368959115097871e-05, 4.4401170148890797e-05, 4.501615603745503e-05, 4.596023096962659e-05, 4.6631475742381485e-05, 4.629068213715892e-05, 4.665962791808387e-05, 4.620217389891745e-05, 4.50537575632015e-05, 4.388144217884204e-05, 4.2441512145229196e-05, 4.319438263914349e-05, 4.321488656444785e-05, 4.13627791509569e-05, 4.0880011599702076e-05, 4.119004141470652e-05, 4.121346075914247e-05, 3.793553722372589e-05, 3.334225939885168e-05, 2.848711638529365e-05, 2.5610176416367805e-05, 2.6178913929526743e-05, 2.706475985365453e-05, 2.8407736090786314e-05, 2.9834995567490964e-05, 3.317256287521774e-05, 3.462661550881519e-05, 3.467244318693981e-05, 3.602471445830099e-05, 3.8257725740223075e-05, 4.052149045092075e-05, 4.174379725700643e-05, 4.3047855480605324e-05, 4.529849973234075e-05, 4.717691583350193e-05, 4.891666565082761e-05, 5.0216531183567755e-05, 5.345251098998578e-05, 5.662826172451792e-05, 6.501991665384056e-05, 6.480875469280191e-05, 6.307988961613232e-05, 5.994596771385808e-05, 5.575647931598516e-05, 5.163576937275355e-05, 4.701847490801693e-05, 4.116311158165234e-05, 3.5775404018020606e-05, 3.1442947289032956e-05, 2.348141618820233e-05, 1.7388956399516275e-05, -7.117158294560965e-07, -1.499855216232365e-05, -3.129368165720296e-05, -5.429856500021924e-05, -7.080599806936725e-05, -7.787593258090296e-05, -8.43591291777029e-05, -9.39524314005636e-05, -0.00010065801262621627, -0.00010266011529664735, -0.00010299865184677785, -0.00010472705097074472, -0.0001039219038964772, -0.00010912186755571771, -0.00011003166152564243, -0.00011181732715819781, -0.00011522552695846223, -0.00011761177030103429, -0.00011500788495872391, -0.00011077093835139071, -0.00010570182716618952, -0.00010127686351659584, -9.792479080452438e-05, -9.538789217743162e-05, -9.40252145165021e-05, -9.189494302209772e-05, -8.96989323034239e-05, -8.875133931376024e-05, -8.802978937840787e-05, -8.659155267288434e-05, -8.26207407101428e-05, -7.835988196488428e-05, -7.48105590096893e-05, -7.104869402840289e-05, -6.81773297430235e-05, -6.341482022929212e-05, -5.82911770245705e-05, -5.475688031157692e-05, -5.488400559940196e-05, -5.6476709251133904e-05, -5.7955754657759e-05, -5.902471858680481e-05, -5.658625518738964e-05, -5.117336707108631e-05, -4.9353542645131514e-05, -4.5690428544281534e-05, -4.17155475215749e-05, -3.6253188640707106e-05, -2.953088311575954e-05, -2.824687958541275e-05, -2.7396767167721714e-05, -2.5680135417055717e-05, -2.3297413706319115e-05, -1.937302528950558e-05, -1.93997614908004e-05, -2.522375773711609e-05, -2.7678441896173287e-05, -3.310776322851541e-05, -3.88061392893573e-05, -4.535365697880528e-05, -4.362805105595965e-05, -3.715275057775566e-05, -2.844353283655891e-05, -2.2826876123671704e-05, -2.127461648216955e-05, -1.936169164935889e-05, -2.1257927821313855e-05, -2.3708387547386226e-05, -2.7244281118460228e-05, -3.2037159157366036e-05, -3.430206916825512e-05, -3.6353963749048084e-05, -3.9658027437748856e-05, -4.3783633031486284e-05, -4.776434169915419e-05, -4.915870070293117e-05, -5.128934045184317e-05, -5.521075498752429e-05, -5.097658507076568e-05, -4.575899483748209e-05, -4.3176199732981925e-05, -4.189310683963749e-05, -4.217070932379632e-05, -4.2427727779944625e-05, -4.454605546000562e-05, -4.574089008930746e-05, -4.623907485432079e-05, -4.205749130508889e-05, -4.0498534209910646e-05, -3.9990766752557105e-05, -3.6571512033549274e-05, -3.379068455989563e-05, -3.254058560423097e-05, -3.15422981234159e-05, -2.955807498891585e-05, -3.0304766717600636e-05, -3.209319091128951e-05, -3.222253576560153e-05, -3.548596775120951e-05, -4.0846851902664705e-05, -4.4275289818718956e-05, -4.5693812111403046e-05, -4.8588691160379746e-05, -5.1834758233953406e-05, -5.495802571770806e-05, -5.736472596138825e-05, -5.9379792654920564e-05, -5.907511519776009e-05, -5.80371023364404e-05, -5.722516369057747e-05, -5.519047076257865e-05, -5.276692252140688e-05, -5.005876051655146e-05, -4.855325907996075e-05, -4.808714223807369e-05, -4.546353132019254e-05, -4.587012137693858e-05, -4.707076585846122e-05, -4.516153500682256e-05, -4.686330735050461e-05, -5.2914671047075385e-05, -5.130636951271299e-05, -5.46548619399973e-05, -5.7390600856510954e-05, -5.8480002097488296e-05, -6.180916395790304e-05, -5.254122172973132e-05, -5.291434524576119e-05, -5.37633222454774e-05, -4.597098277875649e-05, -4.0414107296066817e-05, -3.5619485479459875e-05, -3.584388316921245e-05, -4.1154669708277176e-05, -3.697460138370379e-05, -3.803085801467267e-05, -4.227711413445514e-05, -4.447098068194648e-05, -3.9034949214710874e-05, -4.0934328705137285e-05, -4.687332920090789e-05, -4.527646834787939e-05, -3.998624475249566e-05, -4.149738036123314e-05, -4.6684812308157705e-05, -4.033745872558695e-05, -3.657093140816849e-05, -3.803465538679958e-05, -3.834001694813105e-05, -3.6478023023743446e-05, -4.000832910637305e-05, -4.4603644536135665e-05, -4.564099170863491e-05, -4.5090028709441885e-05, -4.553841019076499e-05, -4.7444752451810856e-05, -4.801027333414737e-05, -4.794547721021124e-05, -4.89202949945446e-05, -4.8198470148920265e-05, -4.73063692517914e-05, -4.772136285302239e-05, -4.922492616131786e-05, -4.768448412823871e-05, -4.3255095905040235e-05, -3.76321811364246e-05, -3.5418779577622615e-05, -3.3169147941286914e-05, -3.073977305073417e-05, -3.204857021507739e-05, -3.287036808873104e-05, -3.362939694473921e-05, -3.6057226033689294e-05, -4.0336664665117454e-05, -4.486206045327062e-05, -4.671402451628089e-05, -4.713955514553065e-05, -4.780310818467114e-05, -4.8173508548220565e-05, -4.9747733690929046e-05, -5.008261935067827e-05, -5.2220917702834946e-05, -5.441355874087539e-05, -5.6203601020299166e-05, -5.766419107262028e-05, -6.024649032215492e-05, -6.251311500749768e-05, -6.488841921945932e-05, -6.434452183445165e-05, -6.303863084264626e-05, -6.548758387692056e-05, -6.453502285413306e-05, -6.5327412677076e-05, -6.825527996242549e-05, -6.865732134108984e-05, -6.708495255425045e-05, -6.91308111956331e-05, -6.740379120149091e-05, -6.878191876922465e-05, -7.091431285290318e-05, -7.123294361765562e-05, -7.536462301160127e-05, -7.756291153744286e-05, -7.732610029251596e-05, -8.15793753470806e-05, -8.397102650832662e-05, -8.491658816774621e-05, -8.578839043294945e-05, -8.733256079781405e-05, -8.486842066888315e-05, -8.644790957217487e-05, -8.849988999766962e-05, -9.026623178473033e-05, -9.338387382249822e-05, -9.611590430391784e-05, -9.901594504996092e-05, -0.00010263127832353043, -0.00010401845262588896, -0.00010518794598401165, -0.00010750314129571247, -0.0001091859062048144, -0.00011087862673549406, -0.00011072154389176845, -0.00011424224107561922, -0.00011700023275044945, -0.00012038770001978024, -0.00012472000879152552, -0.0001230353634744282, -0.00012542165636888455, -0.000125143704901716, -0.0001260753496166562, -0.0001302430811959138, -0.00012991499088590308, -0.00013310022065680976, -0.00013272695608290743, -0.00013279095136047717, -0.00013389478908663334, -0.00013353448901095032, -0.00013316632333743334, -0.00013265707838825365, -0.00013476885972952892, -0.00013621698817685703, -0.0001372421536924468, -0.00014301389074187397, -0.0001412992838757134, -0.00014343600476520425, -0.00013996314719606206, -0.00014387303921949002, -0.00014643768973519525, -0.0001504629247620455, -0.00015319987478763498, -0.00014964162379201582, -0.00014835498538940726, -0.0001481442939982723, -0.0001498358210274396, -0.00015349141385969167, -0.0001565995771998183, -0.0001517331939599807, -0.00015574503240640462, -0.00015008255836354255, -0.00015070799474203, -0.00015650850138743716, -0.00015362795265605843, -0.0001577396361205384, -0.00015624343316454235, -0.00015437910924839045, -0.00015850308419143393, -0.00016316042748992007, -0.00016282536677025277, -0.00017087981061093534, -0.00016851487312190486, -0.00017044209689881415, -0.00017488876703836078, -0.00017447809022820676, -0.00017769493446252713, -0.00017494959711733936, -0.00017463763148841468, -0.00017663787495775793, -0.00017830298293826667, -0.00018296203567376616, -0.00018346205321042515, -0.00018533824394821108, -0.00018592251720613427, -0.00018805134431514503, -0.00019131809869590204, -0.00019060651207947036, -0.00019634437497885552, -0.00019215872404250526, -0.00019411922657720245, -0.0001994947230843881, -0.00019872124249000847, -0.000202048275842493, -0.0002025433433086755, -0.00020177826523234195, -0.0002056030473939358, -0.0002070482348148105, -0.000209823948038881, -0.00020984157780380504, -0.0002063968738744131, -0.00020778742815809632, -0.0002103620195181564, -0.00020926330520532315, -0.0002149070257517106, -0.00021413535899485628, -0.00021509623427709104, -0.0002137203924854825, -0.00021476723159647202, -0.00021396013644539854, -0.0002172436574892095, -0.00022256351491473295, -0.00022267625862672122, -0.00022175465726240133, -0.00022276028027493327, -0.00021979477428086427, -0.00021890264586738606, -0.00022074725756466665, -0.00022116810987599198, -0.00022340126065531244, -0.00022807798226873576, -0.0002286117559298642, -0.00022954643238683826, -0.00022894337446560358, -0.00022540498740638643, -0.00022241860251923174]

ACEvalue_noACE_369_0=ACEvalue_noACE_369_0[0::10]

ACEvalue_noACE_369_1=[5.61695766146464e-06, 3.459547213975192e-06, 5.4777167118961586e-06, 1.661100548777361e-05, 2.917043062269405e-05, 3.761591555554443e-05, 3.1751801525554575e-05, 1.6697181790876287e-05, 3.0593251638494333e-05, 4.0076040063623e-05, 4.069379034063028e-05, 7.371640392519801e-05, -9.617072326722898e-06, -0.00010058480119602874, -0.00010831097547018893, -7.499674932870293e-05, -6.681034301365304e-05, -0.00011754008785827044, -0.00015232263477422145, -0.00015782497465583636, -0.0001413712305381025, -0.0001624182342676031, -0.0001773606066863163, -0.0002074233744997644, -0.00020126183945135147, -0.00022473634092761251, -0.00024465524236096044, -0.00020577306145819305, -0.00019062805779537648, -0.00020824955799324187, -0.00021227135927069603, -0.00021597337764707098, -0.0002267476515961682, -0.0002690599086813526, -0.0002724790619586071, -0.00027246846991952326, -0.000275003462868604, -0.0002727789383602516, -0.0002788357358831321, -0.00026898159859431507, -0.0002633262272500475, -0.00027050618413577325, -0.0002600678584213106, -0.0002735589600199123, -0.000280426831386953]
ACEvalue_noACE_369_2=[9.057770171283599e-06, 1.6869588796243397e-05, 3.17982141755364e-05, 8.297081119863919e-06, -1.2572244617618885e-05, 2.39852394379274e-05, 4.145727279075441e-05, 3.0064060695033667e-05, 2.5470475971630638e-05, -2.6095771812890112e-05, -2.8077266677428362e-05, -1.15431770460417e-05, -3.945561326525723e-05, -0.00010492090200836236, -0.00015744929175506088, -0.0001991176189578084, -0.0002112514693327949, -0.00019021730929856743, -0.0001764413299435709, -0.0001705574009496905, -0.0001621490384318071, -0.0001606411750468665, -0.00021027235050855405, -0.00027653545026414737, -0.0002706685784716318, -0.00029815231915777447, -0.00027527420481861557, -0.0002486138391975966, -0.00033151210176972276, -0.0003157288766742258, -0.0003809737876259469, -0.0003270819316855643, -0.00042037788158547686, -0.0004503154932199191, -0.0004899127644503601, -0.0005123109340697549, -0.0005387200715305881, -0.0005445399818980764, -0.0005220405123548867, -0.0004914325859245581, -0.00046835970602095503, -0.0004539624765498085, -0.00042459059130766486, -0.0004530150394097685, -0.00045261906305568207]
ACEvalue_noACE_369_3=[-8.519382152966482e-06, 4.636832330304729e-06, 6.137365902995456e-06, 1.975269323823954e-05, 3.922809655694102e-05, 6.568317706557458e-05, 7.223054071523863e-05, -6.313292068654928e-05, -0.00010003720958000593, -6.558139459810309e-05, -1.5808125064510958e-05, -4.3604488167951386e-05, -4.993245701694599e-05, -9.209234975273772e-05, -8.216761727845296e-05, -9.593458650138337e-05, -0.00013632309260925947, -0.00011639320456781891, -0.00015235791647882677, -0.0001719116045236653, -0.0001843941574885577, -0.00022204264267546098, -0.00020876860518596728, -0.00021644117414597815, -0.00020656887367125372, -0.00021357713927723724, -0.00022144288443355062, -0.00023366951860768494, -0.0002223723204598835, -0.00023486349518943333, -0.0002452690404064444, -0.0002538303576294434, -0.0002634304087436407, -0.0002794222151505783, -0.0002897035026977807, -0.00030264674388473837, -0.00028234439902636487, -0.0002613615335116913, -0.0002480471734441394, -0.00024245699107488999, -0.00022927434557126763, -0.00023469532614944725, -0.00023345365680137704, -0.00022172448642865676, -0.00019678739647565092]


ACEvalue_noACE_369=[ACEvalue_noACE_369_0,ACEvalue_noACE_369_1,ACEvalue_noACE_369_2,ACEvalue_noACE_369_3]
#--------------433-----------------

test_MSE_noACE_433=[]

#ACEvalue_noACE_433_0=

#ACEvalue_noACE_433_1=
#ACEvalue_noACE_433_2=

#ACEvalue_noACE_433=[ACEvalue_noACE_433_0,ACEvalue_noACE_433_1,ACEvalue_noACE_433_2]

#--------------41-----------------

test_MSE_noACE_41=[]

#ACEvalue_noACE_41_0=
#ACEvalue_noACE_41_1=
#ACEvalue_noACE_41_2=

#ACEvalue_noACE_41=[ACEvalue_noACE_41_0,ACEvalue_noACE_41_1,ACEvalue_noACE_41_2]

#--------------Plotting 369-----------------

var_noACE_369=np.var(ACEvalue_noACE_369, axis=0)
mean_noACE_369=np.mean(ACEvalue_noACE_369, axis=0)

#mean_noACE_369+=var_noACE_369
#print(var.shape)
#print(mean.shape)
x_axis=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450]
#x_axis=np.arange(1, 451)

#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor10_369, axis=0), yerr=np.var(ACEvalue_withACE_factor10_369, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 10")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1_369, axis=0), yerr=np.var(ACEvalue_withACE_factor1_369, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor01_369, axis=0), yerr=np.var(ACEvalue_withACE_factor01_369, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 0.1")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor100_369, axis=0), yerr=np.var(ACEvalue_withACE_factor100_369, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 100")
#plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1000_369, axis=0), yerr=np.var(ACEvalue_withACE_factor1000_369, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1000")
#plt.errorbar(x_axis, mean_noACE_369, yerr=var_noACE_369, linestyle='--', marker='s', markersize=2, label="Without ACE")
plt.plot(x_axis, mean_noACE_369, label="Without ACE")
plt.fill_between(x_axis, mean_noACE_369-var_noACE_369, mean_noACE_369+var_noACE_369)


plt.xlabel('Training epoch')
plt.ylabel('Average causal effect (ACE)')
plt.legend(loc='upper right')
plt.savefig('ACE_entwicklung_369.png')
plt.close()


x_axis=[1,2,3,4,5]
my_xticks = ['Factor 0.1','Factor 1','Factor 10','Factor 100','Factor 1000']
#means_with_ACE_test_MSE_369=[np.mean(test_MSE_withACE_factor01_369),np.mean(test_MSE_withACE_factor1_369),np.mean(test_MSE_withACE_factor10_369),np.mean(test_MSE_withACE_factor100_369),np.mean(test_MSE_withACE_factor1000_369)]
#var_with_ACE_test_MSE_369=[np.var(test_MSE_withACE_factor01_369),np.var(test_MSE_withACE_factor1_369),np.var(test_MSE_withACE_factor10_369),np.var(test_MSE_withACE_factor100_369),np.var(test_MSE_withACE_factor1000_369)]

var_369=np.var(test_MSE_noACE_369)
mean_369=np.mean(test_MSE_noACE_369)
without_ACE_test_MSE_369=[mean_369,mean_369,mean_369,mean_369,mean_369]

plt.xticks(x_axis, my_xticks)
#plt.plot(x_axis,with_ACE_test_MSE, marker='o', markersize=2,label="With ACE")
#plt.plot(x_axis,without_ACE_test_MSE, marker='o', label="Without ACE")
#plt.errorbar(x_axis, means_with_ACE_test_MSE_369, yerr=var_with_ACE_test_MSE_369, linestyle='-', marker='s', markersize=2, label="With ACE")
#plt.errorbar(x_axis, without_ACE_test_MSE_369, yerr=var_369, linestyle='--', marker='s', markersize=2, label="Without ACE")
plt.plot(x_axis, without_ACE_test_MSE_369, label="Without ACE")
plt.fill_between(x_axis, without_ACE_test_MSE_369-var_369, without_ACE_test_MSE_369+var_369, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.xlabel('Factors')
plt.ylabel('Mean Square Error (MSE)')
plt.legend(loc='upper right')
plt.savefig('Test_entwicklung_369.png')
plt.close()

"""

#--------------Plotting 433-----------------


var_noACE_433=np.var(ACEvalue_noACE_433, axis=0)
mean_noACE_433=np.mean(ACEvalue_noACE_433, axis=0)
#print(var.shape)
#print(mean.shape)
x_axis=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450]
#x_axis=np.arange(1, 451)
#plt.plot(x_axis,ACEvalue_withACE_factor1000, label="Factor 1000")
#plt.plot(x_axis,ACEvalue_withACE_factor100, label="Factor 100")
#plt.plot(x_axis,ACEvalue_withACE_factor10, label="Factor 10")
#plt.plot(x_axis,ACEvalue_withACE_factor1, label="Factor 1")
#plt.plot(x_axis,ACEvalue_withACE_factor01, label="Factor 0.1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor10_433, axis=0), yerr=np.var(ACEvalue_withACE_factor10_433, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 10")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1_433, axis=0), yerr=np.var(ACEvalue_withACE_factor1_433, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor01_433, axis=0), yerr=np.var(ACEvalue_withACE_factor01_433, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 0.1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor100_433, axis=0), yerr=np.var(ACEvalue_withACE_factor100_433, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 100")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1000_433, axis=0), yerr=np.var(ACEvalue_withACE_factor1000_433, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1000")
plt.errorbar(x_axis, mean_noACE_433, yerr=var_noACE_433, linestyle='--', marker='s', markersize=2, label="Without ACE")
#plt.plot(x_axis,ACEvalue_noACE_factor1000, label="No ACE")
plt.xlabel('Training epoch')
plt.ylabel('Average causal effect (ACE)')
plt.legend(loc='lower right')
plt.savefig('ACE_entwicklung_433.png')
plt.close()


x_axis=[1,2,3,4,5]
my_xticks = ['Factor 0.1','Factor 1','Factor 10','Factor 100','Factor 1000']
means_with_ACE_test_MSE_433=[np.mean(test_MSE_withACE_factor01_433),np.mean(test_MSE_withACE_factor1_433),np.mean(test_MSE_withACE_factor10_433),np.mean(test_MSE_withACE_factor100_433),np.mean(test_MSE_withACE_factor1000_433)]
var_with_ACE_test_MSE_433=[np.var(test_MSE_withACE_factor01_433),np.var(test_MSE_withACE_factor1_433),np.var(test_MSE_withACE_factor10_433),np.var(test_MSE_withACE_factor100_433),np.var(test_MSE_withACE_factor1000_433)]

var_433=np.var(test_MSE_noACE_433)
mean_433=np.mean(test_MSE_noACE_433)
without_ACE_test_MSE_433=[mean_433,mean_433,mean_433,mean_433,mean_433]

plt.xticks(x_axis, my_xticks)
#plt.plot(x_axis,with_ACE_test_MSE, marker='o', markersize=2,label="With ACE")
#plt.plot(x_axis,without_ACE_test_MSE, marker='o', label="Without ACE")
plt.errorbar(x_axis, means_with_ACE_test_MSE_433, yerr=var_with_ACE_test_MSE_433, linestyle='-', marker='s', markersize=2, label="With ACE")
plt.errorbar(x_axis, without_ACE_test_MSE_433, yerr=var_433, linestyle='--', marker='s', markersize=2, label="Without ACE")
plt.xlabel('Factors')
plt.ylabel('Mean Square Error (MSE)')
plt.legend(loc='upper right')
plt.savefig('Test_entwicklung_433.png')
plt.close()

#--------------Plotting 41-----------------


var_noACE_41=np.var(ACEvalue_noACE_41, axis=0)
mean_noACE_41=np.mean(ACEvalue_noACE_41, axis=0)
#print(var.shape)
#print(mean.shape)
x_axis=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450]
#x_axis=np.arange(1, 451)
#plt.plot(x_axis,ACEvalue_withACE_factor1000, label="Factor 1000")
#plt.plot(x_axis,ACEvalue_withACE_factor100, label="Factor 100")
#plt.plot(x_axis,ACEvalue_withACE_factor10, label="Factor 10")
#plt.plot(x_axis,ACEvalue_withACE_factor1, label="Factor 1")
#plt.plot(x_axis,ACEvalue_withACE_factor01, label="Factor 0.1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor10_41, axis=0), yerr=np.var(ACEvalue_withACE_factor10_41, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 10")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1_41, axis=0), yerr=np.var(ACEvalue_withACE_factor1_41, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor01_41, axis=0), yerr=np.var(ACEvalue_withACE_factor01_41, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 0.1")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor100_41, axis=0), yerr=np.var(ACEvalue_withACE_factor100_41, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 100")
plt.errorbar(x_axis, np.mean(ACEvalue_withACE_factor1000_41, axis=0), yerr=np.var(ACEvalue_withACE_factor1000_41, axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1000")
plt.errorbar(x_axis, mean_noACE_41, yerr=var_noACE_41, linestyle='--', marker='s', markersize=2, label="Without ACE")
#plt.plot(x_axis,ACEvalue_noACE_factor1000, label="No ACE")
plt.xlabel('Training epoch')
plt.ylabel('Average causal effect (ACE)')
plt.legend(loc='lower right')
plt.savefig('ACE_entwicklung_41.png')
plt.close()


x_axis=[1,2,3,4,5]
my_xticks = ['Factor 0.1','Factor 1','Factor 10','Factor 100','Factor 1000']
means_with_ACE_test_MSE_41=[np.mean(test_MSE_withACE_factor01_41),np.mean(test_MSE_withACE_factor1_41),np.mean(test_MSE_withACE_factor10_41),np.mean(test_MSE_withACE_factor100_41),np.mean(test_MSE_withACE_factor1000_41)]
var_with_ACE_test_MSE_41=[np.var(test_MSE_withACE_factor01_41),np.var(test_MSE_withACE_factor1_41),np.var(test_MSE_withACE_factor10_41),np.var(test_MSE_withACE_factor100_41),np.var(test_MSE_withACE_factor1000_41)]

var_41=np.var(test_MSE_noACE_41)
mean_41=np.mean(test_MSE_noACE_41)
without_ACE_test_MSE_41=[mean_41,mean_41,mean_41,mean_41,mean_41]

plt.xticks(x_axis, my_xticks)
#plt.plot(x_axis,with_ACE_test_MSE, marker='o', markersize=2,label="With ACE")
#plt.plot(x_axis,without_ACE_test_MSE, marker='o', label="Without ACE")
plt.errorbar(x_axis, means_with_ACE_test_MSE_41, yerr=var_with_ACE_test_MSE_41, linestyle='-', marker='s', markersize=2, label="With ACE")
plt.errorbar(x_axis, without_ACE_test_MSE_41, yerr=var_41, linestyle='--', marker='s', markersize=2, label="Without ACE")
plt.xlabel('Factors')
plt.ylabel('Mean Square Error (MSE)')
plt.legend(loc='upper right')
plt.savefig('Test_entwicklung_41.png')
plt.close()


#---------Plotting OVERALL---------


x_axis=[1,2,3,4,5]

plt.xticks(x_axis, my_xticks)
#plt.plot(x_axis,with_ACE_test_MSE, marker='o', markersize=2,label="With ACE")
#plt.plot(x_axis,without_ACE_test_MSE, marker='o', label="Without ACE")
plt.errorbar(x_axis, means_with_ACE_test_MSE_41, yerr=var_with_ACE_test_MSE_41, linestyle='-', marker='s', markersize=2, label="With ACE 41")
plt.errorbar(x_axis, without_ACE_test_MSE_41, yerr=var_41, linestyle='--', marker='s', markersize=2, label="Without ACE 41")
plt.errorbar(x_axis, means_with_ACE_test_MSE_433, yerr=var_with_ACE_test_MSE_433, linestyle='-', marker='s', markersize=2, label="With ACE 433")
plt.errorbar(x_axis, without_ACE_test_MSE_433, yerr=var_433, linestyle='--', marker='s', markersize=2, label="Without ACE 433")
plt.errorbar(x_axis, means_with_ACE_test_MSE_369, yerr=var_with_ACE_test_MSE_369, linestyle='-', marker='s', markersize=2, label="With ACE 369")
plt.errorbar(x_axis, without_ACE_test_MSE_369, yerr=var_369, linestyle='--', marker='s', markersize=2, label="Without ACE 369")

mean_overall_withACE=np.mean([means_with_ACE_test_MSE_41,means_with_ACE_test_MSE_433,means_with_ACE_test_MSE_369],axis=0)
var_overall_withACE=np.mean([var_with_ACE_test_MSE_41,var_with_ACE_test_MSE_433,var_with_ACE_test_MSE_369],axis=0)

mean_overall_noACE=np.mean([without_ACE_test_MSE_41,without_ACE_test_MSE_433,without_ACE_test_MSE_369],axis=0)
var_overall_noACE=np.mean([var_41,var_433,var_369],axis=0)

plt.errorbar(x_axis, mean_overall_withACE, yerr=var_overall_withACE, linestyle='-', marker='s', markersize=2, label="With ACE")
plt.errorbar(x_axis, mean_overall_noACE, yerr=var_overall_noACE, linestyle='--', marker='s', markersize=2, label="Without ACE")


plt.xlabel('Factors')
plt.ylabel('Mean Square Error (MSE)')
plt.legend(loc='upper right')
plt.savefig('Test_entwicklung_overall.png')
plt.close()



x_axis=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450]

#mean_01_369=
#mean_1_369=
#mean_10_369=
#mean_100_369=
#mean_1000_369=

#var_01_369=
#var_1_369=
#var_10_369=
#var_100_369=
#var_1000_369=

#mean_01_433=
#mean_1_433=
#mean_10_433=
#mean_100_433=
#mean_1000_433=

#var_01_433=
#var_1_433=
#var_10_433=
#var_100_433=
#var_1000_433=

#mean_10=
#var_10=

plt.errorbar(x_axis, np.mean([np.mean(ACEvalue_withACE_factor10_433, axis=0),np.mean(ACEvalue_withACE_factor10_369, axis=0),np.mean(ACEvalue_withACE_factor10_41, axis=0)], axis=0), yerr=np.mean([np.var(ACEvalue_withACE_factor10_433, axis=0),np.var(ACEvalue_withACE_factor10_369, axis=0),np.var(ACEvalue_withACE_factor10_41, axis=0)], axis=0), linestyle='-', marker='s', markersize=2, label="Factor 10")
plt.errorbar(x_axis, np.mean([np.mean(ACEvalue_withACE_factor1_433, axis=0),np.mean(ACEvalue_withACE_factor1_369, axis=0),np.mean(ACEvalue_withACE_factor1_41, axis=0)], axis=0), yerr=np.mean([np.var(ACEvalue_withACE_factor1_433, axis=0),np.var(ACEvalue_withACE_factor1_369, axis=0),np.var(ACEvalue_withACE_factor1_41, axis=0)], axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1")
plt.errorbar(x_axis, np.mean([np.mean(ACEvalue_withACE_factor01_433, axis=0),np.mean(ACEvalue_withACE_factor01_369, axis=0),np.mean(ACEvalue_withACE_factor01_41, axis=0)], axis=0), yerr=np.mean([np.var(ACEvalue_withACE_factor01_433, axis=0),np.var(ACEvalue_withACE_factor01_369, axis=0),np.var(ACEvalue_withACE_factor01_41, axis=0)], axis=0), linestyle='-', marker='s', markersize=2, label="Factor 0.1")
plt.errorbar(x_axis, np.mean([np.mean(ACEvalue_withACE_factor100_433, axis=0),np.mean(ACEvalue_withACE_factor100_369, axis=0),np.mean(ACEvalue_withACE_factor100_41, axis=0)], axis=0), yerr=np.mean([np.var(ACEvalue_withACE_factor100_433, axis=0),np.var(ACEvalue_withACE_factor100_369, axis=0),np.var(ACEvalue_withACE_factor100_41, axis=0)], axis=0), linestyle='-', marker='s', markersize=2, label="Factor 100")
plt.errorbar(x_axis, np.mean([np.mean(ACEvalue_withACE_factor1000_433, axis=0),np.mean(ACEvalue_withACE_factor1000_369, axis=0),np.mean(ACEvalue_withACE_factor1000_41, axis=0)], axis=0), yerr=np.mean([np.var(ACEvalue_withACE_factor1000_433, axis=0),np.var(ACEvalue_withACE_factor1000_369, axis=0),np.var(ACEvalue_withACE_factor1000_41, axis=0)], axis=0), linestyle='-', marker='s', markersize=2, label="Factor 1000")
plt.errorbar(x_axis, np.mean([mean_noACE_433,mean_noACE_369], axis=0), yerr=np.mean([var_noACE_433, var_noACE_369],axis=0), linestyle='--', marker='s', markersize=2, label="Without ACE")

#var_noACE_41
#mean_noACE_41

#plt.plot(x_axis,ACEvalue_noACE_factor1000, label="No ACE")
plt.xlabel('Training epoch')
plt.ylabel('Average causal effect (ACE)')
plt.legend(loc='lower left')
plt.savefig('ACE_entwicklung_Overall.png')
plt.close()
"""
