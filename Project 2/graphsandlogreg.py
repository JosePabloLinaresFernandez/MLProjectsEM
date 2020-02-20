################################
# Code written by Jose Linares #
# jose.linares386@gmail.com    #
################################

'''
This code is used to make the plots of the statistical analysis
that was done in the report and for doing the logistic regression.

For plotting, most of the code was based in 
https://www.kaggle.com/pavanraj159/predicting-pulsar-star-in-the-universe/notebook

For the logistic regression, some parts where based on the notebooks found in the 
repository of the Machine Learning course for EMJMD in nuclear physics
https://github.com/CompPhysics/MLErasmus

To do the graphs, uncomment each section.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# shuffle random seed generator
np.random.seed(2020) 

########################################
# Reading the data and data management #
########################################
data = pd.read_csv("pulsar_stars.csv")
data.head()

######################
# Correlation matrix #
######################
'''
corr_matrix = data.corr()
plt.figure(figsize=(15,7))
sns.heatmap(corr_matrix,annot=True,linewidth=2,edgecolor='k')
plt.title('Correlation matrix')
# figure needs to be corrected manually to save
plt.savefig('corrmatrix.png')
plt.show()
'''

##############
# Pair plots #
##############
'''
sns.pairplot(data,hue="Target class", corner='True')
plt.title("pair plot for variables")
plt.savefig('pairplot.png',bbox_inches='tight')
plt.show()
'''

###############
# Violin plot #
###############
'''
columns = [x for x in data.columns if x not in ["Target class"]]
length  = len(columns)

plt.figure(figsize=(13,25))

for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/4,j+1)
    sns.violinplot(x=data["Target class"],y=data[i],
                   palette=["Orangered","lime"],alpha=.5)
    plt.title(i)
plt.savefig('violinplot.png',bbox_inches='tight')
plt.show()
'''

##############################################
# Logistic regression  with cross-validation #
##############################################

# We define the hyperparameters to be used
nlambdas = 500 # why not? the code runs relatively fast
lmbdas = np.logspace(-5, 5, nlambdas)
kfold = KFold(n_splits = 5) #cross validation spliting


# We preallocate data 
# true values that will be found later
train_accuracy=np.zeros(lmbdas.shape,np.float64)
test_accuracy=np.zeros(lmbdas.shape,np.float64)
train_red_accuracy=np.zeros(lmbdas.shape,np.float64)
test_red_accuracy=np.zeros(lmbdas.shape,np.float64)
# dummy arrays to be averaged later on
train_accuracy_d=np.zeros(5,np.float64)
test_accuracy_d=np.zeros(5,np.float64)
train_red_accuracy_d=np.zeros(5,np.float64)
test_red_accuracy_d=np.zeros(5,np.float64)

# We create the design matrix X and separate the labels into Y
x_fea = [x for x in data.columns if x not in ["Target class"]]
X = np.zeros((data.shape[0],data.shape[1]-1))
X_red = np.zeros((data.shape[0],3))
Y = np.zeros(data.shape[0])
for i,feature in enumerate(x_fea): # Here we just take the variables of interest
	X[:,i] = data[feature]
	if ' Mean profile' == feature:
		X_red[:,0]
	if 'Kurtosis profile' == feature:
		X_red[:,1]
	if ' Skewness profile' == feature: 
		X_red[:,2]
Y[:] = data['Target class']
'''
# We perform a logistic regression for each value of lambda
for i,lmbda in enumerate(lmbdas):
	#define model
	logreg = LogisticRegression(C=1.0/lmbda,solver='liblinear')

	# Perform the cross-validation
	j = 0
	for train_inds, test_inds in kfold.split(X):
		# Do the split
		X_train = X[train_inds]
		X_red_train = X_red[train_inds]
		Y_train = Y[train_inds]

		X_test = X[test_inds]
		X_red_test = X_red[test_inds]
		Y_test = Y[test_inds]
		
		# We will scale the data
		scaler = StandardScaler()
		scaler.fit(X_train)
		# first on full data
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		# then rescale and do on reduced data
		scaler.fit(X_red_train)
		X_red_train = scaler.transform(X_red_train)
		X_red_test = scaler.transform(X_red_test)
		del scaler
		
		# calculate accuracies for the k fold
		logreg.fit(X_train, Y_train)
		train_accuracy_d[j]=logreg.score(X_train,Y_train)
		test_accuracy_d[j]=logreg.score(X_test,Y_test)

		logreg.fit(X_red_train, Y_train)
		train_red_accuracy_d[j]=logreg.score(X_red_train,Y_train)
		test_red_accuracy_d[j]=logreg.score(X_red_test,Y_test)
		j += 1
		del X_red_train,X_red_test,X_train,Y_train,X_test,Y_test # delete useless data
	#Average to get accuracy values
	train_accuracy[i]=np.mean(train_accuracy_d)
	test_accuracy[i]=np.mean(test_accuracy_d)
	train_red_accuracy[i]=np.mean(train_red_accuracy_d)
	test_red_accuracy[i]=np.mean(test_red_accuracy_d)
	
	print((i+1)/5,'% done')

#plot
plt.semilogx(lmbdas,train_accuracy,label='train')
plt.semilogx(lmbdas,test_accuracy,label='test')
plt.semilogx(lmbdas,train_red_accuracy,label='train reduced') 
#train and test differ very little so to see the different lines we use '--'
plt.semilogx(lmbdas,test_red_accuracy,'--',label='test reduced')
plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{accuracy}$')
plt.grid()
plt.legend()
plt.savefig('logreg.png',bbox_inches='tight')
plt.show()
'''
####################################
# confusion matrix for lambda=10^i #
####################################

lmbdas = np.logspace(-6, 5, 12)

# we do the split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8)
# We will scale the data
scaler = StandardScaler()
scaler.fit(X_train)
# first on full data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
del scaler

# We perform a logistic regression for each value of lambda
fig, ax = plt.subplots(4,3)
for i,lmbda in enumerate(lmbdas):
	#define model
	logreg = LogisticRegression(C=1.0/lmbda,solver='liblinear')
	logreg.fit(X_train, Y_train)
	Y_pred = logreg.predict(X_test)
	#plot
	sns.heatmap(confusion_matrix(Y_test,Y_pred),annot=True,fmt = "d",linecolor="k",linewidths=3,ax=ax[i%4,int(i/4)%3],cbar=False)
	title = '$\\lambda$ = '+str(lmbda)
	ax[i%4,int(i/4)%3].set_title(title)
del X_train,Y_train,X_test,Y_test # delete useless data
#figure needs to be tweaked manually to get a nicely nonoverlapped picture
#plt.savefig('confusionlogreg.png')
plt.show()

