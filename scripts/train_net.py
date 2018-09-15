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
    model.add(torus_transform_layer((11,11),input_shape=(51,51,1)))
    model.add(Convolution2D(space['Convolution2D'], (11, 11), activation=space['activation']))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(torus_transform_layer((5,5)))
    model.add(Convolution2D(space['Convolution2D_1'], (5, 5), activation=space['activation_1']))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(torus_transform_layer((3,3)))
    model.add(Convolution2D(space['Convolution2D_2'], (3, 3), activation=space['activation_2']))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(torus_transform_layer((3,3)))
    model.add(Convolution2D(space['Convolution2D_3'], (3, 3), activation=space['activation_3']))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(torus_transform_layer((3,3)))
    model.add(Convolution2D(space['Convolution2D_4'], (3, 3), activation=space['activation_4']))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(space['Convolution2D_5'], (3, 3), activation=space['activation_5']))
    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(space['Convolution2D_6'], (3, 3), activation=space['activation_6']))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    num_conv = 0#space['activation_7']
    
    for i in range(num_conv):
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(space['Convolution2D_7'], (3, 3), activation=space['activation_8']))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(space['Convolution2D_8'], (3, 3), activation=space['activation_9']))
    
    model.add(Flatten())
    
    model.add(Dense(space['Dense'], activation=space['activation_10']))
    model.add(Dropout(space['Dropout']))
    
    model.add(Dense(space['Dense_1'], activation=space['activation_11']))
    model.add(Dropout(space['Dropout_1']))
    
    num_dense = 0#space['Dropout_2']
    
    for i in range(num_dense):
        model.add(Dense(space['Dense_2'], activation=space['activation_12']))
        model.add(Dropout(space['Dropout_3']))
    
    model.add(Dense(length, activation=linear_bound_above_abs_1))
    
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)

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
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(generator=training_generator,
                    use_multiprocessing=False,
                    workers=8))
    print("Best performing model chosen hyper-parameters:")
    print(best_run) 

    best_run.save('best_model.h5')
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
