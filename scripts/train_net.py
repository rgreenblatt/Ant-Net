import numpy as np
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import DataGenerator
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras import initializers 
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import multi_gpu_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import sys
from torus_transform_layer import torus_transform_layer


#https://gist.github.com/williamFalcon/b03f17991374df99ab371eaeaa7ba610
def create_model(training_generator, testing_generator, length, num_gpus):

    def linear_bound_above_abs_1(x):
        return K.switch(K.less(x, 0), x - 1, x + 1)

    get_custom_objects().update({'linear_bound_above_abs_1': Activation(linear_bound_above_abs_1)})
    model = Sequential()

    kernel_size0 = {{choice([9, 11, 13])}}

    model.add(torus_transform_layer((kernel_size0,kernel_size0),input_shape=(51,51,1)))

    width0 = {{choice([32, 64, 128])}}

    model.add(Convolution2D(width0, (kernel_size0, kernel_size0), activation='linear'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    kernel_size1 = {{choice([3, 5, 7])}}

    model.add(torus_transform_layer((kernel_size1, kernel_size1)))

    width1 = {{choice([16, 32, 64])}}

    model.add(Convolution2D(width1, (kernel_size1, kernel_size1), activation='linear'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    kernel_size2 = {{choice([3, 5])}}

    model.add(torus_transform_layer((kernel_size2,kernel_size2)))

    width2 = {{choice([32, 64])}}

    model.add(Convolution2D(width2, (kernel_size2, kernel_size2), activation='linear'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    kernel_size3 = 3

    model.add(torus_transform_layer((kernel_size3,kernel_size3)))

    width3 = {{choice([64, 128])}}

    model.add(Convolution2D(width3, (kernel_size3, kernel_size3), activation='linear'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    num_conv = 0#{{choice([0, 1])}}

    width4 = 0#{{choice([64, 128])}}
    #
    kernel_size4 = 0#3

    #for i in range(num_conv):
    #	model.add(torus_transform_layer((kernel_size4,kernel_size4)))
    #	model.add(Convolution2D(width4, (kernel_size4, kernel_size4), activation='linear'))
    #	model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    width5 = {{choice([512, 1024])}}

    model.add(Dense(width5, activation='linear'))

    dropout0 = {{uniform(.5, .8)}}

    model.add(Dropout(dropout0))


    width6 = {{choice([512, 1024])}}

    model.add(Dense(width6, activation='linear'))

    dropout1 = {{uniform(.5, .8)}}

    model.add(Dropout(dropout1))

    num_dense = {{choice([0, 1, 2])}}

    width7 = {{choice([512, 1024])}}
    
    dropout2 = {{uniform(.5, .8)}}
    
    for i in range(num_dense):
        model.add(Dense(width7, activation='linear'))
        model.add(Dropout(dropout2))
    
    model.add(Dense(length, activation=linear_bound_above_abs_1))

    learn_r = {{uniform(0.0001, 0.0003)}}
    momentum = {{uniform(0.1, 0.9)}}
    decay = {{uniform(1e-10, 1e-5)}}

    sgd = SGD(lr=learn_r, decay=decay, momentum=momentum, nesterov=True)
    
    print("Model params. w0: ", width0, " w1: ", width1, " w2: ", width2, " w3: ", width3, " w4: ", width4, 
          " w5: ", width5, " w6: ", width6, " w7: ", width7, " k0: ", kernel_size0, " k1: ", kernel_size1, " k2: ",
          kernel_size2, " k3: ", kernel_size3, " k4: ", kernel_size4, " num_conv: ", num_conv, " d0: ", dropout0, 
          " d1: ", dropout1, " d2: ", dropout2, " lr: ", learn_r, " m: ", momentum, " d: ", decay)
    
    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus) 
    
    model.compile(optimizer=sgd, loss='mean_squared_error')
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto', min_delta=0.007)

    #tbCallBack = TensorBoard(log_dir='./graph', write_graph=True, write_images=True)
    
    model.fit_generator(generator=training_generator,
                    validation_data=testing_generator,
                    use_multiprocessing=False,
                    workers=8,
                    epochs=80,
                    callbacks=[earlyStopping]
                    )

    acc = model.evaluate_generator(generator=testing_generator,
                    use_multiprocessing=False,
                    workers=8)
    
    print('Test accuracy:', acc)
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def data():
    length = 6 #This is used as the labels input as that gets provides to the get_label_from_ID func

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
    
    
    y_allowed = False
    
    print("Reading in names list...")
    id_list = pd.read_csv("data_npy/names.txt").values
    
    id_list_train, id_list_test = train_test_split(id_list, test_size=0.20)
    
    id_list_train = np.squeeze(id_list_train)
    id_list_test = np.squeeze(id_list_test)
    
    training_data = {}
    testing_data =  {}
    
    train_id_dict = {}
    test_id_dict = {}
    
    print("Generating dictionary of train labels...")
    #i = 0
    for id_train in id_list_train:
        train_id_dict[id_train] = get_label_from_ID(length, y_allowed, id_train)
        #loaded = np.load('data_npy/' + id_train + '.npy')
        #training_data[id_train] = loaded.reshape(loaded.shape[0], loaded.shape[1], 1)
        #i += 1
    
    print("Generating dictionary of test labels...")
    #i = 0
    for id_test in id_list_test:
        test_id_dict[id_test] = get_label_from_ID(length, y_allowed, id_test)
        #loaded = np.load('data_npy/' + id_test + '.npy')
        #testing_data[id_test] = loaded.reshape(loaded.shape[0], loaded.shape[1], 1)
        #i += 1
    
    
    params = {'dim': (51,51),
              'batch_size': 32,
              'n_channels': 1,
              'y_dim': length,
              'y_dtype': float,
              'shuffle': True}
    
    training_generator = DataGenerator(id_list_train, train_id_dict, data=training_data, **params)
    testing_generator = DataGenerator(id_list_test, test_id_dict, data=testing_data, **params)
    
    num_gpus = 2
    
    if len(sys.argv) > 1:
        num_gpus = int(sys.argv[1])
    
    return training_generator, testing_generator, length, num_gpus

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    training_generator, testing_generator, length, num_gpus = data()

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(generator=training_generator,
                    use_multiprocessing=False,
                    workers=8))
    print("Best performing model chosen hyper-parameters:")
    print(best_run) 

#WAS 0.0007 
#Validate?

#Currently:
#Trained nets for 1-4 x 6 and 1-5 x 6 with working etc
#~20 epochs to overtrain 1-4 x 6 at 0.0007
#? epochs to overtrain 1-5 x 6 at 0.0001

#Next:
#Increase learning rate for 1-5 x 6
#Test wide conv kernal (perhaps 2 * max move)
#Test perceptron style
