# Load libraries
import numpy
import pickle

filename = 'data/model_mlpc_final.sav'
model_mlpc = pickle.load(open(filename, 'rb'))
y_test = numpy.array([[-1, -1, -1, -1, 1, 0, 0, 1, 1]])
print(y_test)
print(model_mlpc.predict(y_test))
