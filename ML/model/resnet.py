from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, BatchNormalization, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers


def resnet_layer(inputs,num_filters = 16, kernel_size = 3,strides = 1,activation ='relu',batch_normalization = True, conv_first = True):

    conv = Conv2D(num_filters, kernel_size = kernel_size, strides = strides,padding ='same', kernel_initializer ='he_normal', kernel_regularizer = l2(1e-4))
    x = inputs
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


  

  
def ResNet(input_shape, depth, num_classes = 36):
    if (depth - 2) % 6 != 0:
        raise ValueError('La produntitat hauria de ser de  6n + 2 (ex. 20, 32, 44 in [a])')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
  
    inputs = Input(shape = input_shape)
    x = resnet_layer(inputs = inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: 
                strides = 2  
            y = resnet_layer(inputs = x,num_filters = num_filters,strides = strides)
            y = resnet_layer(inputs = y,num_filters = num_filters,activation = None)
            if stack > 0 and res_block == 0: 
                x = resnet_layer(inputs = x, num_filters = num_filters, kernel_size = 1,strides = strides,activation = None,batch_normalization = False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
  
 
    x = AveragePooling2D(pool_size = 8)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    y = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation ='softmax',kernel_initializer ='he_normal')(y)
  
    model = Model(inputs = inputs, outputs = outputs)
    return model

       