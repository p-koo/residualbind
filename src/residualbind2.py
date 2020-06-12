from tensorflow import keras

def dilated_residual_block(input_layer, filter_size, activation='relu'):

    num_filters = input_layer.shape.as_list()[-1]  
    nn = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=1,
                                   )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.1)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=2,
                                   )(nn) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.1)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=4,
                                   )(nn) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)


def model():

    # input layer
    inputs = keras.layers.Input(shape=(41,4))

    # layer 1
    nn = keras.layers.Conv1D(filters=96,
                             kernel_size=11,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             )(inputs)                               
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # dilated residual block
    nn = dilated_residual_block(nn, filter_size=3)

    # average pooling
    nn = keras.layers.AveragePooling1D(pool_size=10,  # before it was max pool and pool size of 5 = 0.7834
                                       strides=10, 
                                       padding='same'
                                       )(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Fully-connected NN
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256, activation=None, use_bias=False)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # output layer
    outputs = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
        
    return inputs, outputs
