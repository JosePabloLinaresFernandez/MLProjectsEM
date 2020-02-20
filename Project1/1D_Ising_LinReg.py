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

sns.set(color_codes=True)
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')


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
lambdas = np.logspace(-4, 5, 10) #Lambda values to be used
energies = ising_energies(states,L) # energies for states of L particles
X = create_X(states,L) #create design matrix
boots = 50 #number of bootstraps
X_train, X_test, energies_train, energies_test = train_test_split(X, energies, test_size=0.4) #split

#Initilize R2, MSE, Bias, variance vectors:
train_R2score = {
    "OLS": np.zeros(lambdas.size),
    "Ridge": np.zeros(lambdas.size),
    "LASSO": np.zeros(lambdas.size)
}

test_R2score = {
    "OLS": np.zeros(lambdas.size),
    "Ridge": np.zeros(lambdas.size),
    "LASSO": np.zeros(lambdas.size)
}

MSE = {
    "OLS": np.zeros(lambdas.size),
    "Ridge": np.zeros(lambdas.size),
    "LASSO": np.zeros(lambdas.size)
}

bias = {
    "OLS": np.zeros(lambdas.size),
    "Ridge": np.zeros(lambdas.size),
    "LASSO": np.zeros(lambdas.size)
}

variance = {
    "OLS": np.zeros(lambdas.size),
    "Ridge": np.zeros(lambdas.size),
    "LASSO": np.zeros(lambdas.size)
}


plot_counter = 1

fig = plt.figure(figsize=(32, 54))

for i, _lambda in enumerate(tqdm.tqdm(lambdas)):
	for key, method in zip(
		["OLS", "Ridge", "LASSO"],
		[skl.LinearRegression(), skl.Ridge(alpha=_lambda), skl.Lasso(alpha=_lambda)]
	):
		#method = method.fit(X_train, energies_train)
		########################
		# Boostrap starts here #
		########################

		# first define dummy variables
		energies_pred_test = np.zeros((boots,len(energies_test)),float)
		energies_pred_train = np.zeros((boots,len(energies_train)),float)
		
		#rescaling data
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		# dummy variable for R2 scores
		R2_train = np.zeros(boots,float)
		R2_test = np.zeros(boots,float)

		#train_errors[key][i] = method.score(X_train, energies_train)
		#test_errors[key][i] = method.score(X_test, energies_test)
		for j in range(boots):
			#define a new split for each bootstrap
			X_, energies_ = resample(X_train,energies_train)
			clf = method.fit(X_,energies_)
			energies_pred_test[j,:] = clf.predict(X_test).ravel() #store results for each bootstrap iteration
			R2_train[j] = clf.score(X_,energies_)
			R2_test[j] = clf.score(X_test,energies_test)
		# Compute average values of MSE, bias and variance
		MSE[key][i] = np.mean(np.mean((energies_test - energies_pred_test)**2, axis=0, keepdims=True))
		bias[key][i] = np.mean((energies_test - np.mean(energies_pred_test, axis=0, keepdims=True))**2)
		variance[key][i] = np.mean(np.var(energies_pred_test, axis=0, keepdims=True))
		train_R2score[key][i] = np.mean(R2_train)
		test_R2score[key][i] = np.mean(R2_test)
		# Compute average values of R2
		
		omega = method.coef_.reshape(L, L)
		#print(omega)
		plt.subplot(6, 5, plot_counter)
		plt.imshow(omega, **cmap_args)
		plt.title(r"%s, $\lambda = %.4f$" % (key, _lambda))
		plot_counter += 1
		plt.colorbar()
print(test_R2score)
plt.savefig('multilambdamethod.png')

fig2 = plt.figure(figsize=(20, 14))

colors = {
    "OLS": "r",
    "Ridge": "k",
    "LASSO": "g"
}

for key in train_R2score:
    plt.semilogx(
        lambdas,
        train_R2score[key],
        colors[key],
        label="Train {0}".format(key),
        linewidth=4.0
    )

for key in test_R2score:
    plt.semilogx(
        lambdas,
        test_R2score[key],
        colors[key] + "--",
        label="Test {0}".format(key),
        linewidth=4.0
    )
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"$\lambda$", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.tick_params(labelsize=20)
plt.savefig('MSE_R2.png')

fig3 = plt.figure(figsize=(20, 14))

colors = {
    "OLS": "r",
    "Ridge": "k",
    "LASSO": "g"
}

for key in MSE:
    plt.semilogx(
        lambdas,
        MSE[key],
        colors[key],
        label="Error {0}".format(key),
        linewidth=4.0
    )

for key in bias:
    plt.semilogx(
        lambdas,
        bias[key],
        colors[key] + "--",
        label="Bias {0}".format(key),
        linewidth=4.0
    )
for key in variance:
    plt.semilogx(
        lambdas,
        variance[key],
        colors[key] + ":",
        label="Variance {0}".format(key),
        linewidth=4.0
    )
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"$\lambda$", fontsize=20)
plt.ylabel(r" ", fontsize=20)
plt.tick_params(labelsize=20)
plt.savefig('biasvariance.png')
plt.show()



