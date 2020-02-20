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
train_to_test_ratio=0.2 # training samples

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels


# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

del X_ordered, X_disordered, Y_ordered, Y_disordered
# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio,test_size=1.0-train_to_test_ratio)

# full data set
X_full=np.concatenate((X_critical,X))
Y_full=np.concatenate((Y_critical,Y))
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
train_accuracy=np.zeros(lmbdas.shape,np.float64)
test_accuracy=np.zeros(lmbdas.shape,np.float64)
critical_accuracy=np.zeros(lmbdas.shape,np.float64)

train_accuracy_CV=np.zeros(lmbdas.shape,np.float64)
test_accuracy_CV=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_CV=np.zeros(lmbdas.shape,np.float64)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

#We scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i,lmbda in enumerate(lmbdas):
	logreg = LogisticRegression(C=1.0/lmbda)#solver='liblinear',
	logreg.fit(X_train, Y_train)
	train_accuracy[i]=logreg.score(X_train,Y_train)
	test_accuracy[i]=logreg.score(X_test,Y_test)
	critical_accuracy[i]=logreg.score(X_critical,Y_critical)
	print("Test set accuracy: {:.2f}".format(test_accuracy[i]))
	print("Critical test accuracy: {:.2f}".format(critical_accuracy[i]))
	print("Training set acuracy: {:.2f}".format(train_accuracy[i]))


del X,Y,X_train,Y_train,X_test,Y_test,X_critical,Y_critical
plt.semilogx(lmbdas,train_accuracy,'*-b',label='train')
plt.semilogx(lmbdas,test_accuracy,'*-r',label='test')
plt.semilogx(lmbdas,critical_accuracy,'*-g',label='critical')


plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{accuracy}$')

plt.grid()
plt.legend()
plt.savefig('2Disingscaled.png')
plt.show()
