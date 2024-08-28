
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten
import numpy as np
import keras

keras.backend.clear_session()

n_inputs = 4

d_model = Sequential()
d_model.add(Dense(2, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
d_model.summary()

rd_model = Sequential()
rd_model.add(Dense(4,  activation='relu', input_dim=2))
rd_model.summary()


m_model = Sequential()
m_model.add(d_model)
m_model.add(rd_model)
m_model.summary()
m_model.compile(loss='binary_crossentropy', optimizer='adam')

a = np.ones((1,4))
m_model.fit(a, a, epochs=1000)


