from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class torus_transform_layer(Layer):

    def __init__(self, target_kernel, **kwargs):
        super(torus_transform_layer, self).__init__(**kwargs)
        assert target_kernel[0] == target_kernel[1]
        assert (target_kernel[0] % 2) == 1
        self.scaling = int((target_kernel[0] - 1) / 2)

    def compute_output_shape(self, input_shape):
        length = len(input_shape)

        assert length == 4

        x = input_shape[1] + self.scaling * 2
        y = input_shape[2] + self.scaling * 2

        return (input_shape[0], x, y, input_shape[3]) 
    def call(self, inputs):
        for_left = inputs[:,:,-self.scaling:]
        for_right = inputs[:,:,:self.scaling]
        for_top = inputs[:,-self.scaling:]
        for_bottom = inputs[:,:self.scaling]

        for_top_left = inputs[:,-self.scaling:,-self.scaling:]
        for_top_right = inputs[:,-self.scaling:,:self.scaling]
        for_bottom_left = inputs[:,:self.scaling,-self.scaling:]
        for_bottom_right = inputs[:,:self.scaling,:self.scaling]

        list_top = []
        list_middle = []
        list_bottom = []

        top = K.concatenate([for_top_left, for_top, for_top_right], axis=2)
        middle = K.concatenate([for_left, inputs, for_right], axis=2)
        bottom = K.concatenate([for_bottom_left, for_bottom, for_bottom_right], axis=2)

        outputs = K.concatenate([top, middle, bottom], axis=1)
        return outputs
