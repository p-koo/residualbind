from tensorflow import keras
from keras import backend as K
from scipy import stats
import numpy as np

class ResidualBind():

    def __init__(self, input_shape=(41,4), weights_path='.'):

        self.input_shape = input_shape
        self.weights_path = weights_path
        self.model = self.build(input_shape)

    def build(self, input_shape):
        K.clear_session()

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

        # input layer
        inputs = keras.layers.Input(shape=input_shape)

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
        nn = keras.layers.AveragePooling1D(pool_size=3,  
                                           strides=3, 
                                           padding='same'
                                           )(nn)
        nn = keras.layers.Dropout(0.2)(nn)

        # layer 2
        nn = keras.layers.Conv1D(filters=128,
                                 kernel_size=3,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 )(nn)                               
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = dilated_residual_block(nn, filter_size=3)
        nn = keras.layers.AveragePooling1D(pool_size=4, 
                                           strides=4, 
                                           )(nn)
        nn = keras.layers.Dropout(0.3)(nn)

        # Fully-connected NN
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(256, activation=None, use_bias=False)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        # output layer
        logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
        outputs = keras.layers.Activation('sigmoid')(logits)
                
        return keras.Model(inputs=inputs, outputs=outputs)

    def load_weights(self):
        self.model.load_weights(self.weights_path)
        print('  Loading model from: ' + self.weights_path)

    def save_weights(self):
        self.model.save_weights(self.weights_path)
        print('  Saving model to: ' + self.weights_path)

    def fit(self, train, valid, num_epochs=300, batch_size=100, 
            patience=25, lr=0.001, lr_decay=0.3, decay_patience=7):

        auroc = keras.metrics.AUC(curve='ROC', name='auroc')
        aupr = keras.metrics.AUC(curve='PR', name='aupr')
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy', auroc, aupr])

        es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc', #'val_aupr',#
                                                    patience=patience, 
                                                    verbose=1, 
                                                    mode='max', 
                                                    restore_best_weights=False)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                      factor=lr_decay,
                                                      patience=decay_patience, 
                                                      min_lr=1e-7,
                                                      mode='max',
                                                      verbose=1) 

        history = self.model.fit(train['inputs'], train['targets'], 
                            epochs=num_epochs,
                            batch_size=batch_size, 
                            shuffle=True,
                            validation_data=(valid['inputs'], valid['targets']), 
                            callbacks=[es_callback, reduce_lr])

        # save model
        self.save_weights()


    def test_model(self, test, batch_size=100, weights='best'):
        results = model.evaluate(test['inputs'], test['targets'], batch_size=512)
        return results

    def predict(self, X, batch_size=100, weights='best'):
        if weigths == 'best':
            self.load_weights()
        return self.model.predict(X, batch_size=batch_size)

    def predict_windows(self, X, stride=1, batch_size=100, weights='best'):
        if weigths == 'best':
            self.load_weights()
        L = self.input_shape[0]
        predictions = []
        for i in range(1, len(X)-L, stride):
            predictions.append(self.predict(X[:,i:i+L,:], batch_size, weights))
        return np.concatenate(predictions, axis=1)

