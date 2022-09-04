

from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn


model = load_model('generator_model_100K.h5')


vector = randn(100) 
vector = vector.reshape(1, 100)


X = model.predict(vector)


pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()
