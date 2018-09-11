import numpy as np
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import DataGenerator
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import initializers 
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import multi_gpu_model
import sys

def linear_bound_above_abs_1(x):
    return K.switch(K.less(x, 0), x - 1, x + 1)

get_custom_objects().update({'linear_bound_above_abs_1': Activation(linear_bound_above_abs_1)})

#https://gist.github.com/williamFalcon/b03f17991374df99ab371eaeaa7ba610
def VGG_19(length=6, weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(51,51,1)))
    model.add(Convolution2D(64, (10, 10), activation='linear'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(length, activation=linear_bound_above_abs_1))

    #initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)

    if weights_path:
        model.load_weights(weights_path)

    return model

def get_label_from_ID(length, y_allowed, ID):
    parsed = ID.replace('ant_data__', '')

    index_value = 1
    
    if y_allowed:
        index_value = 2

    states = np.empty((length, index_value), dtype=float)

    for i in range(length):
        vec_string = re.sub(r'__.*', '', parsed)

        first = re.sub(r'_.*', '', vec_string)

        if y_allowed:
            states[i][0] = int(first)

        first += '_'
        second = re.sub(first, '', vec_string)
        if y_allowed:
            states[i][1] = int(second)
        else:
            states[i][0] = int(second)
            

        if(i != length - 1):
            remove_vec = vec_string + "__"
            parsed = re.sub(remove_vec, '', parsed)

    return np.squeeze(states)


length = 6 #This is used as the labels input as that gets provides to the get_label_from_ID func
y_allowed = False

print("Reading in names list...")
id_list = pd.read_csv("data_npy4/names.txt").values

id_list_train, id_list_test = train_test_split(id_list, test_size=0.20)

id_list_train = np.squeeze(id_list_train)
id_list_test = np.squeeze(id_list_test)

train_id_dict = {}
test_id_dict = {}

print("Generating dictionary of train labels...")
for id_train in id_list_train:
    train_id_dict[id_train] = get_label_from_ID(length, y_allowed, id_train)

print("Generating dictionary of test labels...")
for id_test in id_list_test:
    test_id_dict[id_test] = get_label_from_ID(length, y_allowed, id_test)


params = {'dim': (51,51),
          'batch_size': 64,
          'n_channels': 1,
          'y_dim': length,
          'y_dtype': float,
          'shuffle': True}

training_generator = DataGenerator(id_list_train, train_id_dict, **params)
testing_generator = DataGenerator(id_list_test, test_id_dict, **params)

model = VGG_19(length)


#WAS 0.0007 
#Validate?
sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)

num_gpus = 2

if len(sys.args) > 1:
    num_gpus = sys.args[1]

model = multi_gpu_model(model, gpus=num_gpus) 

model.compile(optimizer=sgd, loss='mean_squared_error')

tbCallBack = TensorBoard(log_dir='./graph', write_graph=True, write_images=True)

model.fit_generator(generator=training_generator,
                    validation_data=testing_generator,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=50,
                    callbacks=[tbCallBack]
                    )

model.save("models/initial.hdf5")

#Currently:
#Trained nets for 1-4 x 6 and 1-5 x 6 with working etc
#~20 epochs to overtrain 1-4 x 6 at 0.0007
#? epochs to overtrain 1-5 x 6 at 0.0001

#Next:
#Increase learning rate for 1-5 x 6
#Test wide conv kernal (perhaps 2 * max move)
#Test perceptron style
