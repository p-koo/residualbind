from tensorflow import keras
from tensorflow.keras import backend as K
from scipy import stats
import numpy as np
import itertools

class ResidualBind():

    def __init__(self, input_shape=(41,4), num_class=1, weights_path='.', classification=False):

        self.input_shape = input_shape
        self.num_class = num_class
        self.weights_path = weights_path
        self.classification = classification
        self.model = self.build(input_shape)


    def build(self, input_shape):
        K.clear_session()

        def residual_block(input_layer, filter_size, activation='relu', dilated=False):

            if dilated:
                factor = [2, 4, 8]
            else:
                factor = [1]
            num_filters = input_layer.shape.as_list()[-1]  

            nn = keras.layers.Conv1D(filters=num_filters,
                                           kernel_size=filter_size,
                                           activation=None,
                                           use_bias=False,
                                           padding='same',
                                           dilation_rate=1,
                                           )(input_layer) 
            nn = keras.layers.BatchNormalization()(nn)
            for f in factor:
                nn = keras.layers.Activation('relu')(nn)
                nn = keras.layers.Dropout(0.1)(nn)
                nn = keras.layers.Conv1D(filters=num_filters,
                                               kernel_size=filter_size,
                                               strides=1,
                                               activation=None,
                                               use_bias=False, 
                                               padding='same',
                                               dilation_rate=f,
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
        nn = residual_block(nn, filter_size=3, dilated=True)

        # average pooling
        nn = keras.layers.AveragePooling1D(pool_size=10)(nn)
        nn = keras.layers.Dropout(0.2)(nn)

        """
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
        nn = residual_block(nn, filter_size=3, dilated=False)
        
        nn = keras.layers.AveragePooling1D(pool_size=4, 
                                           strides=4, 
                                           )(nn)
        nn = keras.layers.Dropout(0.3)(nn)
        """
        # Fully-connected NN
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(256, activation=None, use_bias=False)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        # output layer
        outputs = keras.layers.Dense(self.num_class, activation='linear', use_bias=True)(nn)
        
        if self.classification:
            outputs = keras.layers.Activation('sigmoid')(outputs)

        return keras.Model(inputs=inputs, outputs=outputs)

    def load_weights(self):
        self.model.load_weights(self.weights_path)
        print('  Loading model from: ' + self.weights_path)

    def save_weights(self):
        self.model.save_weights(self.weights_path)
        print('  Saving model to: ' + self.weights_path)

    def _compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
            
        # set up optimizer and metrics
        if not self.classification:
            self.model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
        else:
            model.compile(optimizer=optimizer,
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy', auroc, aupr])
        

    def fit(self, train, valid, num_epochs=300, batch_size=100, 
            patience=25, lr=0.001, lr_decay=0.3, decay_patience=7):

        self._compile_model()

        if self.classification:
            self._fit_classification(train, valid, num_epochs, batch_size, 
            patience, lr, lr_decay, decay_patience)
        else:
            self._fit_regression(train, valid, num_epochs, batch_size, 
            patience, lr, lr_decay, decay_patience)


    def _fit_regression(self, train, valid, num_epochs=300, batch_size=100, 
            patience=25, lr=0.001, lr_decay=0.3, decay_patience=7):

        # fit model with decaying learning rate and store model with highest Pearson r
        best_pearsonr = 0
        counter = 0
        decay_counter = 0
        for epoch in range(num_epochs):
            print('Epoch %d out of %d'%(epoch, num_epochs))

            # training epoch
            history = self.model.fit(train['inputs'], train['targets'], 
                                             epochs=1,
                                             batch_size=batch_size, 
                                             shuffle=True)

            # get metrics on validation set
            predictions = self.model.predict(valid['inputs'], batch_size=batch_size)
            corr = pearsonr_scores(valid['targets'], predictions)
            print('  Validation: ' + str(np.mean(corr)))

            # check for early stopping and decay learning rate conditions
            if best_pearsonr < corr:
                best_pearsonr = corr
                decay_counter = 0
                counter = 0
                self.save_weights()
            else:
                counter += 1
                decay_counter += 1
                if decay_counter == decay_patience:
                    lr *= lr_decay
                    lr = np.maximum(lr, 1e-6)
                    K.set_value(self.model.optimizer.lr, lr)
                    decay_counter = 0
                    print('  Decaying learning rate to: %f'%(lr))

                if counter == patience:
                    print('  Patience ran out... Early Stopping!')
                    break


    def _fit_classification(self, train, valid, num_epochs=300, batch_size=100, 
            patience=25, lr=0.001, lr_decay=0.3, decay_patience=7):

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

        # fit model
        history = self.model.fit(x_train, y_train, 
                                epochs=num_epochs,
                                batch_size=batch_size, 
                                shuffle=True,
                                validation_data=(valid['inputs'], valid['targets']), 
                                callbacks=[es_callback, reduce_lr])

        # save weights
        weights_path = os.path.join(params_path, name+'.hdf5')
        model.save_weights(weights_path)


    def test_model(self, test, batch_size=100, load_weights=None):

        if self.classification:
            metrics = self.model.test_model(test['inputs'], test['targets'])
        else:
            predictions = self.predict(test['inputs'], batch_size, load_weights)
            metrics = pearsonr_scores(test['targets'], predictions)
        return metrics

    def predict(self, X, batch_size=100, load_weights=False):
        if load_weights:
            self.load_weights()

        return self.model.predict(X, batch_size=batch_size)

    def predict_windows(self, X, stride=1, batch_size=100, load_weights=False):
        if load_weights:
            self.load_weights()

        L = self.input_shape[0]
        predictions = []
        for i in range(1, X.shape[1]-L, stride):
            predictions.append(self.predict(X[:,i:i+L,:], batch_size, weights=False))
        return np.hstack(predictions)







#-------------------------------------------------------------------------------------


class GlobalImportance():
    def __init__(self, residualbind, alphabet='ACGU'):
        self.residualbind = residualbind
        self.alphabet = alphabet
        self.x_null = None
        self.x_null_index = None

    def set_x_null(self, x_null):
        # x_null should be one-hot 
        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)


    def set_null_model(self, seq_model, num_sim=1000):
        factor = 2

        # sequence length
        L = seq_model.shape[0]

        x_null = np.zeros((num_sim*factor, L, 4))
        for n in range(num_sim*factor):

            # generate uniform random number for each nucleotide in sequence
            Z = np.random.uniform(0,1,L)

            # calculate cumulative sum of the probabilities
            cum_prob = seq_model.cumsum(axis=1)

            # find bin that matches random number for each position
            for l in range(L):
                index=[j for j in range(4) if Z[l] < cum_prob[l,j]][0]
                x_null[n,l,index] = 1

        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)
        self.predict_null()
        #self.filter_null_model(low=10, high=90, num_sim=num_sim)


    def filter_null_model(self, low=10, high=90, num_sim=1000):

        high = np.percentile(self.null_scores, high)
        low = np.percentile(self.null_scores, low)
        index = np.where((self.null_scores < high)&(self.null_scores > low))[0]
        self.set_x_null(self.x_null[index][:num_sim])
        self.predict_null()

               
    def predict_null(self, class_index=0):
        self.null_scores = self.residualbind.predict(self.x_null)[:, class_index]
        self.mean_null_score = np.mean(self.null_scores)


    def embed_patterns(self, patterns):
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)
        for pattern, position in patterns:

            # convert pattern to categorical representation
            pattern_index = np.array([self.alphabet.index(i) for i in pattern])

            # embed pattern 
            x_index[:,position:position+len(pattern)] = pattern_index

        # convert to categorical representation to one-hot 
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n,l,a] = 1.0

        return one_hot
    

    def set_hairpin_null(self, stem_left=7, stem_right=23, stem_size=9):
        one_hot = np.copy(self.x_null)
        stem_left_end = stem_left + stem_size
        stem_right_end = stem_right + stem_size
        rc = one_hot[:,stem_left:stem_left_end,:]
        rc = rc[:,:,::-1]
        rc = rc[:,::-1,:]
        one_hot[:,stem_right:stem_right_end,:] = rc
        self.set_x_null(one_hot)

    
    def embed_pattern_hairpin(self, patterns, stem_left=7, stem_right=23, stem_size=9):
        
        # set the null to be a stem-loop
        self.set_hairpin_null(stem_left=7, stem_right=23, stem_size=9)
        
        # embed the pattern
        one_hot = self.embed_patterns(patterns)
        
        # fix the step
        stem_left_end = stem_left + stem_size
        stem_right_end = stem_right + stem_size
        rc = one_hot[:,stem_left:stem_left_end,:]
        rc = rc[:,:,::-1]
        rc = rc[:,::-1,:]
        one_hot[:,stem_right:stem_right_end,:] = rc

        return  one_hot



    def embed_predict_effect(self, patterns, class_index=0):
        one_hot = self.embed_patterns(patterns)
        return self.residualbind.predict(one_hot)[:, class_index] - self.null_scores


    def predict_effect(self, one_hot, class_index=0):
        predictions = self.residualbind.predict(one_hot)[:, class_index]
        return predictions - self.null_scores


    def optimal_kmer(self, kmer_size=7, position=17, class_index=0):
        
        # generate all kmers             
        kmers = ["".join(p) for p in itertools.product(list(self.alphabet), repeat=kmer_size)]

        # score each kmer
        mean_scores = []
        for i, kmer in enumerate(kmers):
            if np.mod(i+1,500) == 0:
                print("%d out of %d"%(i+1, len(kmers)))
            
            effect = self.embed_predict_effect((kmer, position), class_index)
            mean_scores.append(np.mean(effect))

        kmers = np.array(kmers)
        mean_scores = np.array(mean_scores)

        # sort by highest prediction
        sort_index = np.argsort(mean_scores)[::-1]

        return kmers[sort_index], mean_scores[sort_index]


    def kmer_mutagenesis(self, kmer='UGCAUG', position=17, class_index=0):
        
        # get wt score
        wt_score = np.mean(self.embed_predict_effect((kmer, position), class_index))

        # score each mutation
        L = len(kmer)
        A = len(self.alphabet)
        mean_scores = np.zeros((L, A))
        for l in range(L):
            for a in range(A):
                if kmer[l] == self.alphabet[a]:
                    mean_scores[l,a] = wt_score

                else:
                    # introduce mutation
                    mut_kmer = list(kmer)
                    mut_kmer[l] = self.alphabet[a]
                    mut_kmer = "".join(mut_kmer)
                                
                    # score mutant
                    mean_scores[l,a]  = np.mean(self.embed_predict_effect((mut_kmer, position), class_index))

        return mean_scores



    def positional_bias(self, motif='UGCAUG', positions=[2, 12, 23, 33], class_index=0):

        # loop over positions and measure effect size of intervention
        all_scores = []
        for position in positions:
            all_scores.append(self.embed_predict_effect((motif, position), class_index))

        return np.array(all_scores)



    def multiple_sites(self, motif='UGCAUG', positions=[17, 10, 25, 3], class_index=0):

        # loop over positions and measure effect size of intervention
        all_scores = []
        for i, position in enumerate(positions):

            # embed motif multiple times
            interventions = []
            for j in range(i+1):
                interventions.append((motif, positions[j]))

            all_scores.append(self.embed_predict_effect(interventions, class_index))

        return np.array(all_scores)


    def gc_bias(self, motif='UGCAUG', motif_position=17,
                gc_motif='GCGCGC', gc_positions=[34, 2], class_index=0):

        all_scores = []


        # background sequence with gc-bias on right side
        all_scores.append(self.embed_predict_effect((gc_motif, gc_positions[0]), class_index))

        # background sequence with motif at center
        all_scores.append(self.embed_predict_effect((motif, motif_position), class_index))

        # create interventions for gc bias
        for position in gc_positions:

            interventions = [(motif, motif_position), (gc_motif, position)]
            all_scores.append(self.embed_predict_effect(interventions, class_index))

        return np.array(all_scores)



#-------------------------------------------------------------------------------------


def pearsonr_scores(y_true, y_pred, mask_value=None):
    corr = []
    for i in range(y_true.shape[1]):
        if mask_value:
            index = np.where(y_true[:,i] != mask_value)[0]
            corr.append(stats.pearsonr(y_true[index,i], y_pred[index,i])[0])
        else:
            corr.append(stats.pearsonr(y_true[:,i], y_pred[:,i])[0])
    return np.array(corr)

