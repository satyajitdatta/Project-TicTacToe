# Load libraries
import numpy
import pickle

filename = 'data/model_knn_single.sav'
model_knn = pickle.load(open(filename, 'rb'))
y_test = numpy.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])
print(y_test)
print(model_knn.predict(y_test))
