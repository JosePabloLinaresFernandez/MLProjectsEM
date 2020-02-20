import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy.linalg as scl
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
import tqdm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
def create_X(states, L):
	# Number of elements in beta
	X = np.ones((len(states),L**2),float)
	for E in range(len(states)):
		for i in range(L):
			q = int(i*L)
			for j in range(L):
				X[E,q+j] = states[E,i]*states[E,j]
	return(X)

def ising_energies(states,L):
	"""
	This function calculates the energies of the states in the nn Ising Hamiltonian
	"""
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L]-=1.0
		# compute energies
		E = np.einsum('...i,ij,...j->...',states,J,states)
	return(E)
	# calculate Ising energies
	energies=ising_energies(states,L)

###########################
# Ising model definitions #
###########################

np.random.seed(2018)
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

#################################### 
# Define Ising model parameters    #
# system size->L                   #
# create 10000 random Ising states #
####################################
L=40
states=np.random.choice([-1, 1], size=(10000,L))

#########################
# Parameters to be used #
# in general            #
#########################
energies = ising_energies(states,L) # energies for states of L particles
X = create_X(states,L) #create design matrix
X_train, X_test, energies_train, energies_test = train_test_split(X, energies, test_size=0.4) #split
# Then we can use the stadndard scale to scale our data 

##########################
# Neural network solving #
##########################

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
n_hidden_neurons = 50
epochs = 100
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver='adam')
        dnn.fit(X_train, energies_train)
        DNN_scikit[i][j] = dnn
        '''J = dnn.fit(X_train,energies_train).coef_.reshape(L,L)
		plt.subplots()
		plt.imshow(J)
		plt.show()'''
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, energies_test))
        print()

import seaborn as sns

sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
	for j in range(len(lmbd_vals)):
		dnn = DNN_scikit[i][j]
        
		train_pred = dnn.predict(X_train) 
		test_pred = dnn.predict(X_test)
		TrainA = r2_score(energies_train, train_pred)
		if TrainA > 0.:
			train_accuracy[i][j] = TrainA
		else:
			train_accuracy[i][j] = 0
		TestA = r2_score(energies_test, test_pred)
		if TestA > 0:
			test_accuracy[i][j] = TestA
		else:
			test_accuracy[i][j] = 0
      
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, annot_kws={"size": 18}, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy",fontsize=18)
ax.set_ylabel("$\eta$",fontsize=18)
ax.set_yticklabels(eta_vals)
ax.set_xlabel("$\lambda$",fontsize=18)
ax.set_xticklabels(lmbd_vals)
plt.tick_params(labelsize=18)
#plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, annot_kws={"size": 18}, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy",fontsize=18)
ax.set_ylabel("$\eta$",fontsize=18)
ax.set_yticklabels(eta_vals)
ax.set_xlabel("$\lambda$",fontsize=18)
ax.set_xticklabels(lmbd_vals)
plt.tick_params(labelsize=18)
plt.show()
