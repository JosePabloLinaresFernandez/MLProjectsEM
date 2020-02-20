import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

# Reading data using PANDA 
data = pd.read_csv("pulsar_stars.csv")
data.head()

#DATA
x_fea = [x for x in data.columns if x not in ['target_class']]
X = np.zeros((data.shape[0],data.shape[1]-1))
X_red = np.zeros((data.shape[0],3))
Y = np.zeros(data.shape[0])
for i,feature in enumerate(x_fea): # Here we just take the variables of interest
	X[:,i] = data[feature]
	if ' Mean of the integrated profile' == feature:
		X_red[:,0]
	if ' Excess kurtosis of the integrated profile' == feature:
		X_red[:,1]
	if ' Skewness of the integrated profile' == feature: 
		X_red[:,2]
Y[:] = data['target_class']
np.random.seed(2030)

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 66)

# The maximum depth of the tree:
max_degree=20
maxdepth=np.zeros(max_degree)
#Initialize accuracy arrays for training and test data
train_accuracy_gini=np.zeros(max_degree)
test_accuracy_gini=np.zeros(max_degree)
train_accuracy_entropy=np.zeros(max_degree)
test_accuracy_entropy=np.zeros(max_degree)
train_var_gini=np.zeros(max_degree,np.float64)
test_var_gini=np.zeros(max_degree,np.float64)
train_var_entropy=np.zeros(max_degree,np.float64)
test_var_entropy=np.zeros(max_degree,np.float64)

train_accuracy_gini_red=np.zeros(max_degree)
test_accuracy_gini_red=np.zeros(max_degree)
train_accuracy_entropy_red=np.zeros(max_degree)
test_accuracy_entropy_red=np.zeros(max_degree)
train_var_gini_red=np.zeros(max_degree,np.float64)
test_var_gini_red=np.zeros(max_degree,np.float64)
train_var_entropy_red=np.zeros(max_degree,np.float64)
test_var_entropy_red=np.zeros(max_degree,np.float64)

k = 5
# dummy arrays to be averaged later on
train_accuracy_gini_d=np.zeros(k,np.float64)
test_accuracy_gini_d=np.zeros(k,np.float64)
train_accuracy_entropy_d=np.zeros(k,np.float64)
test_accuracy_entropy_d=np.zeros(k,np.float64)

train_accuracy_gini_red_d=np.zeros(k,np.float64)
test_accuracy_gini_red_d=np.zeros(k,np.float64)
train_accuracy_entropy_red_d=np.zeros(k,np.float64)
test_accuracy_entropy_red_d=np.zeros(k,np.float64)


# Number of kfold for CV
kfold = KFold(n_splits = k)

#Decision Tree
for i in range(1,max_degree+1):
    maxdepth[i-1]=i
    tree_gini = DecisionTreeClassifier(max_depth=i)
    tree_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=i)
    # Perform the cross-validation
    j = 0
    for train_inds, test_inds in kfold.split(X):
        # Do the split
        X_train = X[train_inds]
        Y_train = Y[train_inds]
        X_test = X[test_inds]
        Y_test = Y[test_inds]

        X_train_red = X_red[train_inds]
        X_test_red = X_red[test_inds]

        # We will scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        # first on full data
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # rescale for reduced data
        scaler.fit(X_train_red)
        X_train_red = scaler.transform(X_train_red)
        X_test_red = scaler.transform(X_test_red)
        # calculate accuracies for the k fold
        tree_gini.fit(X_train, Y_train)
        train_accuracy_gini_d[j]=tree_gini.score(X_train,Y_train)
        test_accuracy_gini_d[j]=tree_gini.score(X_test,Y_test)
        tree_entropy.fit(X_train, Y_train)
        train_accuracy_entropy_d[j]=tree_entropy.score(X_train,Y_train)
        test_accuracy_entropy_d[j]=tree_entropy.score(X_test,Y_test)

        tree_gini.fit(X_train_red, Y_train)
        train_accuracy_gini_red_d[j]=tree_gini.score(X_train_red,Y_train)
        test_accuracy_gini_red_d[j]=tree_gini.score(X_test_red,Y_test)
        tree_entropy.fit(X_train_red, Y_train)
        train_accuracy_entropy_red_d[j]=tree_entropy.score(X_train_red,Y_train)
        test_accuracy_entropy_red_d[j]=tree_entropy.score(X_test_red,Y_test)
        j += 1

    print(test_accuracy_gini_d,'gini',i)
    print(test_accuracy_entropy_d,'entropy',i)
    print(' ')

    train_accuracy_gini[i-1]=np.mean(train_accuracy_gini_d)
    test_accuracy_gini[i-1]=np.mean(test_accuracy_gini_d)
    train_accuracy_entropy[i-1]=np.mean(train_accuracy_entropy_d)
    test_accuracy_entropy[i-1]=np.mean(test_accuracy_entropy_d)

    train_var_gini[i-1]=np.std(train_accuracy_gini_d)
    test_var_gini[i-1]=np.std(test_accuracy_gini_d)
    train_var_entropy[i-1]=np.std(train_accuracy_entropy_d)
    test_var_entropy[i-1]=np.std(test_accuracy_entropy_d)

    train_accuracy_gini_red[i-1]=np.mean(train_accuracy_gini_red_d)
    test_accuracy_gini_red[i-1]=np.mean(test_accuracy_gini_red_d)
    train_accuracy_entropy_red[i-1]=np.mean(train_accuracy_entropy_red_d)
    test_accuracy_entropy_red[i-1]=np.mean(test_accuracy_entropy_red_d)

    train_var_gini_red[i-1]=np.std(train_accuracy_gini_red_d)

    test_var_gini_red[i-1]=np.std(test_accuracy_gini_red_d)
    train_var_entropy_red[i-1]=np.std(train_accuracy_entropy_red_d)
    test_var_entropy_red[i-1]=np.std(test_accuracy_entropy_red_d)

#Plot the accuracy for training and test data as a function of max_depth
fig, tree=plt.subplots()
tree.set_xlabel('max_depth')
tree.set_ylabel('accuracy')
tree.plot(maxdepth,train_accuracy_gini, color='r', label='Training [Gini]' )
tree.plot(maxdepth,test_accuracy_gini, color='r',linestyle="--", label='Test [Gini]' )
tree.plot(maxdepth,train_accuracy_entropy, color='b', label='Training [entropy]' )
tree.plot(maxdepth,test_accuracy_entropy, color='b',linestyle="--", label='Test [entropy]' )
#tree.plot(maxdepth,train_accuracy_gini_red, color='y', label='Training reduced [Gini]' )
#tree.plot(maxdepth,test_accuracy_gini_red, color='y',linestyle="--", label='Test reduced [Gini]' )
#tree.plot(maxdepth,train_accuracy_entropy_red, color='g', label='Training reduced [entropy]' )
#tree.plot(maxdepth,test_accuracy_entropy_red, color='g',linestyle="--", label='Test reduced [entropy]' )
tree.legend()
#plor the variance for training and test data as a function of max_depth
fig, treevar=plt.subplots()
treevar.set_xlabel('max_depth')
treevar.set_ylabel('accuracy standard deviation')
treevar.plot(maxdepth,train_var_gini, color='r', label='Training [Gini]' )
treevar.plot(maxdepth,test_var_gini, color='r',linestyle="--", label='Test [Gini]' )
treevar.plot(maxdepth,train_var_entropy, color='b', label='Training [entropy]' )
treevar.plot(maxdepth,test_var_entropy, color='b',linestyle="--", label='Test [entropy]' )
#treevar.plot(maxdepth,train_var_gini_red, color='y', label='Training reduced [Gini]' )
#treevar.plot(maxdepth,test_var_gini_red, color='y',linestyle="--", label='Test reduced [Gini]' )
#treevar.plot(maxdepth,train_var_entropy_red, color='g', label='Training reduced [entropy]' )
#treevar.plot(maxdepth,test_var_entropy_red, color='g',linestyle="--", label='Test reduced [entropy]' )
treevar.legend()

plt.show()




