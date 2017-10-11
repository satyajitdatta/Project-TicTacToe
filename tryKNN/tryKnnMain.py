# Load libraries
import numpy
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

A = numpy.loadtxt('data/tictac_final.txt')
X = A[:, :9]
Y = A[:, 9:]

model_knn = KNeighborsClassifier()
valdation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=valdation_size)
print(type(X_train))
print(type(Y_train))
model_knn.fit(X_train, Y_train.ravel())

Y_prediction = model_knn.predict(X_validation)
print(Y_prediction)

print(accuracy_score(Y_validation, Y_prediction))
print(confusion_matrix(Y_validation, Y_prediction))
print(classification_report(Y_validation, Y_prediction))
