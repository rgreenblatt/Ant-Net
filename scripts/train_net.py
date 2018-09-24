import numpy as np
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from generator import DataGenerator
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import initializers 
from keras.layers import Activation, BatchNormalization, Input
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import sys
from torus_transform_layer import torus_transform_layer
from keras.models import load_model
from keras.models import Model
import keras


#https://gist.github.com/williamFalcon/b03f17991374df99ab371eaeaa7ba610
def create_model(training_generator, testing_generator, length, num_gpus, weight_path, save_path):
    
    np.random.seed(seed=2642)

    def not_quite_linear(x):
        return K.tanh(x / 5.0) * 5.0

    def linear_bound_above_abs_1(x):
        return K.switch(K.less(not_quite_linear(x), 0), x - 1, x + 1)
    
    get_custom_objects().update({'linear_bound_above_abs_1': Activation(linear_bound_above_abs_1)})
    get_custom_objects().update({'not_quite_linear': Activation(not_quite_linear)})
    #model = Sequential()

    kernel_size_0 = {{choice([7, 9, 11])}}

    
    def resnet_layer(inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
    
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Convolution2D number of filters
            kernel_size (int): Convolution2D square kernel dimensions
            strides (int): Convolution2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
    
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Convolution2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
    
        x = inputs
        
        if kernel_size != 1:
            x  = torus_transform_layer((kernel_size, kernel_size))(x)
        
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v2(input_shape, depth, num_classes):
        """ResNet Version 2 Model builder [b]
    
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Convolution2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Convolution2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
    
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
    
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)
    
        inputs = Input(shape=input_shape)
        # v2 performs Convolution2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters_in,
                         kernel_size=11,
                         conv_first=True)
    
        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                this_kernal = 3
                activation = not_quite_linear
                batch_normalization = False
                strides = 1
                if stage == 0:
                    this_kernal = 7
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample
    
                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 kernel_size=this_kernal,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                    #x = resnet_layer(inputs=x,
                    #                 num_filters=num_filters_out,
                    #                 kernel_size=1,
                    #                 strides=strides,
                    #                 activation=None,
                    #                 batch_normalization=False)
                x = keras.layers.add([x, y])
    
            num_filters_in = num_filters_out
    
        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = Activation(not_quite_linear)(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation=linear_bound_above_abs_1,
                        kernel_initializer='he_normal')(y)
    
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model





    #model.add(torus_transform_layer((11,11),input_shape=(51,51,1)))
    #model.add(Convolution2D(16, (11, 11), activation=not_quite_linear))

    #model.add(torus_transform_layer((11,11)))
    #model.add(Convolution2D(16, (11, 11), activation=not_quite_linear))

    #model.add(torus_transform_layer((9,9)))
    #model.add(Convolution2D(16, (9, 9), activation=not_quite_linear))

    #model.add(torus_transform_layer((9, 9)))
    #model.add(Convolution2D(16, (9, 9), activation=not_quite_linear))

    #model.add(torus_transform_layer((3, 3)))
    #model.add(MaxPooling2D((3,3), strides=(2,2)))

    #model.add(torus_transform_layer((7,7)))
    #model.add(Convolution2D(32, (7, 7), activation=not_quite_linear))

    #model.add(torus_transform_layer((7,7)))
    #model.add(Convolution2D(32, (7, 7), activation=not_quite_linear))

    #model.add(torus_transform_layer((5,5)))
    #model.add(Convolution2D(32, (5, 5), activation=not_quite_linear))

    #model.add(torus_transform_layer((5,5)))
    #model.add(Convolution2D(32, (5, 5), activation=not_quite_linear))

    #model.add(torus_transform_layer((3, 3)))
    #model.add(MaxPooling2D((3,3), strides=(2,2)))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(64, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3, 3)))
    #model.add(MaxPooling2D((3,3), strides=(2,2)))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3,3)))
    #model.add(Convolution2D(128, (3, 3), activation=not_quite_linear))

    #model.add(torus_transform_layer((3, 3)))
    #model.add(MaxPooling2D((3,3), strides=(2,2)))

    #model.add(Flatten())

    #model.add(Dense(256, activation=not_quite_linear))
    #model.add(Dropout(0.4))

    #model.add(Dense(256, activation=not_quite_linear))
    #model.add(Dropout(0.4))

    #model.add(Dense(length, activation=linear_bound_above_abs_1))
   
    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    n = 3
    depth = n * 9 + 2


    model = resnet_v2((51,51,1), depth, length)

 
    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus) 

    use_amsgrad = {{choice([True])}}
    

    if save_path != None:
        model = load_model(save_path)
    
    model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['accuracy'])
    
    #earlyStopping=EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='auto', min_delta=0.007)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    #tbCallBack = TensorBoard(log_dir='./graph', write_graph=True, write_images=True)
    
    model.fit_generator(generator=training_generator,
                    validation_data=testing_generator,
                    use_multiprocessing=True,
                    workers=8,
                    epochs=80,
                    callbacks=callbacks
                    )
    
    model.save('model_initial.h5')
    
    acc = model.evaluate_generator(generator=testing_generator,
                    use_multiprocessing=True,
                    workers=8)
    
    print('Test accuracy:', acc)
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def data():

    np.random.seed(seed=263342)

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
              'batch_size': 256,
              'n_channels': 1,
              'y_dim': length,
              'y_dtype': float,
              'shuffle': True}
    
    training_generator = DataGenerator(id_list_train, train_id_dict, data=training_data, **params)
    testing_generator = DataGenerator(id_list_test, test_id_dict, data=testing_data, **params)
    
    num_gpus = 2
    
    if len(sys.argv) > 1:
        num_gpus = int(sys.argv[1])
    
    save_path = None

    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    
    return training_generator, testing_generator, length, num_gpus, save_path

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())
    training_generator, testing_generator, length, num_gpus, weight_path, save_path = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(generator=training_generator,
                    use_multiprocessing=True,
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
