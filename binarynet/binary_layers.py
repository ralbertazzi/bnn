from keras import backend as K

from keras.layers import Dense, Conv2D
from keras import constraints
from keras import initializers

from binary_ops import binarize


'''Binarized Dense and Convolution2D layers
References: 
"BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
'''


class Clip(constraints.Constraint):

    def __call__(self, p):
        return K.clip(p, -1., 1.)


class BinaryDense(Dense):

    def __init__(self, units, **kwargs):
        super(BinaryDense, self).__init__(units, use_bias=False, 
                                          kernel_initializer=initializers.RandomUniform(-1., 1.),
                                          kernel_constraint=Clip(), **kwargs)


    def call(self, inputs):
        
        # 1. Binarize weights
        binary_kernel = binarize(self.kernel, H=1.)
        
        # 2. Perform matrix multiplication
        output = K.dot(inputs, binary_kernel)
        return output


class BinaryConv2D(Conv2D):

    def __init__(self, filters, kernel_size, **kwargs):
        super(BinaryConv2D, self).__init__(filters, kernel_size, use_bias=False, padding='same',
                                           kernel_initializer=initializers.RandomUniform(-1., 1.),
                                           kernel_constraint=Clip(), **kwargs)
        
    def call(self, inputs):
        
        # 1. Binarize weights
        binary_kernel = binarize(self.kernel, H=1.)
        
        # 2. Perform convolution
        outputs = K.conv2d(
            inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        
        return outputs


# Aliases
BinaryConvolution2D = BinaryConv2D
