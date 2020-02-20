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
from sklearn.model_selection import cross_validate
warnings.filterwarnings("ignore")

# Reading data using PANDA 
data = pd.read_csv("pulsar_stars.csv")
data.head()

#DATA
targets = data["target_class"]
features = data.drop("target_class", axis = 1)
np.random.seed(2018)

#Split data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 66)

#Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

# The maximum depth of the tree:
max_degree=40
maxdepth=np.zeros(max_degree)
#Initialize accuracy arrays for training and test data
train_accuracy_gini=np.zeros(max_degree)
test_accuracy_gini=np.zeros(max_degree)
train_accuracy_entropy=np.zeros(max_degree)
test_accuracy_entropy=np.zeros(max_degree)
train_accuracy_forest_gini=np.zeros(max_degree)
test_accuracy_forest_gini=np.zeros(max_degree)
train_accuracy_forest_entropy=np.zeros(max_degree)
test_accuracy_forest_entropy=np.zeros(max_degree)

#Decision Tree  and Random Forest without cross validation
for i in range(1,max_degree+1):
    maxdepth[i-1]=i
    #Decision tree method with gini criteria
    tree_gini = DecisionTreeClassifier(max_depth=i)
    tree_gini.fit(X_train, y_train)
    train_accuracy_gini[i-1]=tree_gini.score(X_train, y_train)
    test_accuracy_gini[i-1]=tree_gini.score(X_test, y_test)
    #Decision tree method with gini criteria
    tree_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=i)
    tree_entropy.fit(X_train, y_train)
    train_accuracy_entropy[i-1]=tree_entropy.score(X_train, y_train)
    test_accuracy_entropy[i-1]=tree_entropy.score(X_test, y_test)
    # Random Forest Method with 100 trees
    forest_gini=RandomForestClassifier(n_estimators=100, max_depth=i)
    forest_gini.fit(X_train, y_train)
    train_accuracy_forest_gini[i-1]=forest_gini.score(X_train, y_train)
    test_accuracy_forest_gini[i-1]=forest_gini.score(X_test, y_test)
    forest_entropy=RandomForestClassifier(n_estimators=100, max_depth=i)
    forest_entropy.fit(X_train, y_train)
    train_accuracy_forest_entropy[i-1]=forest_entropy.score(X_train, y_train)
    test_accuracy_forest_entropy[i-1]=forest_entropy.score(X_test, y_test)
print("accuracy on training data with criterion gini [tree]=",np.around(train_accuracy_gini,decimals=2))
print("accuracy on test data with criterion gini[tree]=", np.around(test_accuracy_gini,decimals=2))
print("accuracy on training data with criterion entropy [tree]=",np.around(train_accuracy_entropy,decimals=2))
print("accuracy on test data with criterion entrpy [tree]=", np.around(test_accuracy_entropy,decimals=2))
print("accuracy on training data with criterion gini [forest]=",np.around(train_accuracy_forest_gini,decimals=2))
print("accuracy on test data with criterion gini [forest] =", np.around(test_accuracy_forest_gini,decimals=2))
print("accuracy on training data with criterion entropy [forest]=",np.around(train_accuracy_forest_entropy,decimals=2))
print("accuracy on test data with criterion entrpy [forest]=", np.around(test_accuracy_forest_entropy,decimals=2))
#Plot the accuracy for training and test data as a function of max_depth
fig, tree=plt.subplots()
tree.set_xlabel('max_depth')
tree.set_ylabel('accuracy')
tree.set_title('Decision Tree')
tree.plot(maxdepth,train_accuracy_gini, color='r', label='Training [Gini]' )
tree.plot(maxdepth,test_accuracy_gini, color='r',linestyle="--", label='Test [Gini]' )
tree.plot(maxdepth,train_accuracy_entropy, color='b', label='Training [entropy]' )
tree.plot(maxdepth,test_accuracy_entropy, color='b',linestyle="--", label='Test [entropy]' )
tree.legend()

fig2, forest=plt.subplots()
forest.set_xlabel('max_depth')
forest.set_ylabel('accuracy')
forest.set_title('Random Forest')
forest.plot(maxdepth,train_accuracy_forest_gini, color='r', label='Training [Gini]' )
forest.plot(maxdepth,test_accuracy_forest_gini, color='r',linestyle="--", label='Test [Gini]' )
forest.plot(maxdepth,train_accuracy_forest_entropy, color='b', label='Training [entropy]' )
forest.plot(maxdepth,test_accuracy_forest_entropy, color='b',linestyle="--", label='Test [entropy]' )
forest.legend()
plt.show()

