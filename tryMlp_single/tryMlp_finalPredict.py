# Load libraries
import numpy
import pickle

filename = 'data/model_mlpc_single.sav'
model_mlpc = pickle.load(open(filename, 'rb'))
y_test = numpy.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]])
print(y_test)
print(model_mlpc.predict(y_test))
