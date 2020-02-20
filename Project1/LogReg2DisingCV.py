import numpy as np
import pickle, os
from urllib.request import urlopen 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import KFold

#Comment this to turn on warnings
#warnings.filterwarnings('ignore')


# shuffle random seed generator
np.random.seed(2020) 


##########################
# Ising model parameters #
##########################

L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit

#######################
# Loading the dataset #
#######################

# url to data
url_main = 'https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/';

# LOAD DATA; the data consists of 16*10000 samples taken 
# in T=np.arange(0.25,4.0001,0.25):
data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
# The labels are obtained from the following file:
label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"


########
# DATA #
########

# pickle reads the file and returns the Python object (1D array, compressed bits)
data = pickle.load(urlopen(url_main + data_file_name)) 
# Decompress array and reshape for convenience
data = np.unpackbits(data).reshape(-1, 1600) 
data=data.astype('int')
# map 0 state to -1 (Ising variable can take values +/-1)
data[np.where(data==0)]=-1 


##########
# LABELS #
##########

# (convention is 1 for ordered states and 0 for disordered states)
# pickle reads the file and returns the Python object (here just a 1D 
# array with the binary labels)
labels = pickle.load(urlopen(url_main + label_file_name)) 

###########################################
# Contructing the train and test datasets #
###########################################

X = data
Y = labels

num_classes=2
train_to_test_ratio=0.04 # training samples

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]
print(len(X_ordered),len(X_disordered))
del data,labels


# define training and test data sets
#X=np.concatenate((X_ordered,X_disordered))
#Y=np.concatenate((Y_ordered,Y_disordered))

# full data set
#X_full=np.concatenate((X_critical,X))
#Y_full=np.concatenate((Y_critical,Y))
lmbdas = np.logspace(-5,5,11) 
###########################
# Visualizing some states #
###########################
'''
# set colourbar map
cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].imshow(X_ordered[40001].reshape(L,L),**cmap_args)
axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=10)
axarr[0].tick_params(labelsize=10)

axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=10)
axarr[1].tick_params(labelsize=10)

im=axarr[2].imshow(X_disordered[30001].reshape(L,L),**cmap_args)
axarr[2].set_title('$\\mathrm{disordered\\ phase}$',fontsize=10)
axarr[2].tick_params(labelsize=10)

#fig.subplots_adjust(right=2.0)

plt.show()
'''
####################
# preallocate data #
####################
train_accuracy_var=np.zeros(lmbdas.shape,np.float64)
test_accuracy_var=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_var=np.zeros(lmbdas.shape,np.float64)

train_accuracy_CV=np.zeros(lmbdas.shape,np.float64)
test_accuracy_CV=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_CV=np.zeros(lmbdas.shape,np.float64)

# loop over regularisation strength for non cross-validated attempt
# If this part is ran, then take out the loop on the plotting section

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

for i,lmbda in enumerate(lmbdas):
	# create dummy arrays to be averaged later
	print(i/len(lmbdas) *100,'%')
	logreg = LogisticRegression(C=1.0/lmbda)
	#prealocate data
	train_accuracy_d = np.zeros(k,float)
	test_accuracy_d = np.zeros(k,float) 
	critical_accuracy_d = np.zeros(k,float)
	counter = 0
	for train_inds, test_inds in kfold.split(X_ordered):
		countercondition = 0
		for train_inds_dis, test_inds_dis in kfold.split(X_disordered):
			if counter == countercondition: #some trick we used to get 
				# swap training for test so that the machine can handle the algorithm
				X_train_ord = X_ordered[test_inds]
				Y_train_ord = Y_ordered[test_inds]

				X_train_dis = X_disordered[test_inds_dis]
				Y_train_dis = Y_disordered[test_inds_dis]				
				
				
				X_test_ord = X_ordered[train_inds]
				Y_test_ord = Y_ordered[train_inds]
				
				X_test_dis = X_disordered[train_inds_dis]
				Y_test_dis = Y_disordered[train_inds_dis]
				
				# join the data so that we can train the model
				X_train = np.concatenate((X_train_ord,X_train_dis))
				Y_train = np.concatenate((Y_train_ord,Y_train_dis))

				X_test = np.concatenate((X_test_ord,X_test_dis))
				Y_test = np.concatenate((Y_test_ord,Y_test_dis))

				#We scale our data
				scaler = StandardScaler()
				scaler.fit(X_train)
				X_train = scaler.transform(X_train)
				X_test = scaler.transform(X_test)

				
				logreg.fit(X_train, Y_train) #do the fit
				train_accuracy_d[counter]=logreg.score(X_train,Y_train) #store each value of accuracy
				test_accuracy_d[counter]=logreg.score(X_test,Y_test)
				critical_accuracy_d[counter]=logreg.score(X_critical,Y_critical)
				print(counter)
			countercondition += 1
		counter += 1
		
	#We store the mean values of the accuracies
	train_accuracy_CV[i] = np.mean(train_accuracy_d)
	test_accuracy_CV[i] = np.mean(test_accuracy_d)
	critical_accuracy_CV[i] = np.mean(critical_accuracy_d)
	print(train_accuracy_d)
	#We store the variance of the accuracies
	train_accuracy_var[i] = np.var(train_accuracy_d)
	test_accuracy_var[i] = np.var(test_accuracy_d)
	critical_accuracy_var[i] = np.var(critical_accuracy_d)

	del train_accuracy_d, test_accuracy_d, critical_accuracy_d

	print("Test set accuracy with CV: {:.2f}".format(test_accuracy_CV[i]))
	print("Critical test accuracy with CV: {:.2f}".format(critical_accuracy_CV[i]))
	print("Training set acuracy with CV: {:.2f}".format(train_accuracy_CV[i]))


del X,Y,X_train,Y_train,X_test,Y_test,X_critical,Y_critical

#plt.semilogx(lmbdas,train_accuracy_CV,'*--b',label='CV train')
#plt.semilogx(lmbdas,test_accuracy_CV,'*--r',label='CV test')
#plt.semilogx(lmbdas,critical_accuracy_CV,'*--g',label='CV critical')

#plt.xlabel('$\\lambda$')
#plt.ylabel('$\\mathrm{accuracy}$')

#plt.grid()
#plt.legend()
#plt.savefig('2DisingscaledCV.png')


plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{Variance}$')
plt.semilogx(lmbdas,train_accuracy_var,'*--b',label='CV train')
plt.semilogx(lmbdas,test_accuracy_var,'*--r',label='CV test')
plt.semilogx(lmbdas,critical_accuracy_var,'*--g',label='CV critical')
plt.grid()
plt.legend()
plt.savefig('2Disingscaledvariance.png')
plt.show()


