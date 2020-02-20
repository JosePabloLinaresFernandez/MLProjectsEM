import numpy as np
import pickle, os
from urllib.request import urlopen 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score

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

num_classes=2																	
train_to_test_ratio=0.2 # training samples

# naming some variables so that it makes more sense
# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio,test_size=1.0-train_to_test_ratio)

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
n_hidden_neurons = 50
epochs = 10
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver='adam')
        dnn.fit(X_train, Y_train)
        
        DNN_scikit[i][j] = dnn
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, Y_test))
        print()

import seaborn as sns

sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
critical_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
	for j in range(len(lmbd_vals)):
		dnn = DNN_scikit[i][j]
        
		train_pred = dnn.predict(X_train) 
		test_pred = dnn.predict(X_test)
		critical_pred = dnn.predict(X_critical)
		train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
		test_accuracy[i][j] = accuracy_score(Y_test, test_pred)
		critical_accuracy[i][j] = accuracy_score(Y_critical,critical_pred)

      
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis",xticklabels=lmbd_vals,yticklabels=eta_vals)
ax.set_title("Training Accuracy",size=26)
ax.set_ylabel("$\eta$",size=26)
ax.set_xlabel("$\lambda$",size=26)
#plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis",xticklabels=lmbd_vals,yticklabels=eta_vals)
ax.set_title("Test Accuracy",size=26)
ax.set_ylabel("$\eta$",size=26)
ax.set_xlabel("$\lambda$",size=26)
#plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(critical_accuracy, annot=True, ax=ax, cmap="viridis",xticklabels=lmbd_vals,yticklabels=eta_vals)
ax.set_title("Critical Accuracy",size=26)
ax.set_ylabel("$\eta$",size=26)
ax.set_xlabel("$\lambda$",size=26)
plt.show()




