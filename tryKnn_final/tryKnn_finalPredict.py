# Load libraries
import numpy
import pickle

filename = 'data/model_knn_final.sav'
model_knn = pickle.load(open(filename, 'rb'))
y_test = numpy.array([[-1, -1, -1, -1, 1, 0, 0, 1, 1]])
print(y_test)
print(model_knn.predict(y_test))
