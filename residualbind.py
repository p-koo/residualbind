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
        outputs = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
                
        return keras.Model(inputs=inputs, outputs=outputs)

    def load_weights(self):
        self.model.load_weights(self.weights_path)
        print('  Loading model from: ' + self.weights_path)

    def save_weights(self):
        self.model.save_weights(self.weights_path)
        print('  Saving model to: ' + self.weights_path)

    def fit(self, train, valid, num_epochs=300, batch_size=100, 
            patience=25, lr=0.001, lr_decay=0.3, decay_patience=7):

        # set up optimizer and metrics
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

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

    def test_model(self, test, batch_size=100, weights=None):
        predictions = self.predict(test['inputs'], batch_size, weights)
        corr = pearsonr_scores(test['targets'], predictions)
        return corr

    def predict(self, X, batch_size=100, weights=None):
        if weights == 'best':
            self.load_weights()
        return self.model.predict(X, batch_size=batch_size)

    def predict_windows(self, X, stride=1, batch_size=100, weights=None):
        if weights == 'best':
            self.load_weights()
        L = self.input_shape[0]
        predictions = []
        for i in range(1, len(X)-L, stride):
            predictions.append(self.predict(X[:,i:i+L,:], batch_size, weights))
        return np.concatenate(predictions, axis=1)



class GlobalImportance():
    def __init__(self, residualbind, class_index=0, alphabet='ACGU'):
        self.residualbind = residualbind
        self.class_index = class_index
        self.alphabet = alphabet


    def set_x_null(self, x_null):
        # x_null should be one-hot 
        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)


    def set_null_model(self, seq_model, num_sim=1000):

        # sequence length
        L = seq_model.shape[0]

        x_null = np.zeros((num_sim, L, 4))
        for n in range(num_sim):

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

                
    def predict_null(self):
        self.null_scores = self.residualbind.predict(self.x_null)[:, self.class_index]
        self.mean_null_score = np.mean(self.null_scores)


    def embed_patterns(self, patterns):
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)
        for pattern, position in patterns:

            # convert pattern to categorical representation
            pattern_index = np.array([alphabet.index(i) for i in pattern])

            # embed pattern 
            x_index[:,position:position+len(pattern)] = pattern

        # convert to categorical representation to one-hot 
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n,l,a] = 1.0

        return one_hot
    
    def predict_effect(self, patterns, weights=None):
        one_hot = self.embed_patterns(patterns, weights)
        predictions = self.residualbind.predict(one_hot, )[:, self.class_index]
        return predictions - self.null_scores



    def optimal_kmer(self, kmer_size=7, position=17):
        
        # generate all kmers             
        kmers = ["".join(p) for p in itertools.product(list(self.alphabet), repeat=kmer_size)]

        # score each kmer
        mean_scores = []
        for i, kmer in enumerate(kmers):
            if np.mod(i+1,500) == 0:
                print("%d out of %d"%(i+1, len(kmers)))
            
            effect = self.predict_effect((kmer, position))
            mean_scores.append(np.mean(effect))

        kmers = np.array(kmers)
        mean_scores = np.array(mean_scores)

        # sort by highest prediction
        sort_index = np.argsort(mean_scores)[::-1]

        return kmers[sort_index], mean_scores[sort_index]


    def kmer_mutagenesis(self, motif='UGCAUG', position=17):
        
        # get wt score
        wt_score = glo.predict_effect((motif, position))

        # score each mutation
        L = len(motif)
        A = len(self.alphabet)
        mean_scores = np.zeros((L, A))
        for l in range(L):
            for a in range(A):
                if motif[l] == self.alphabet[a]:
                    mean_scores[l,a] = wt_score

                else:
                    # introduce mutation
                    mut_motif = list(motif)
                    mut_motif[l] = self.alphabet[a]
                    mut_motif = "".join(mut_motif)
                                
                    # score mutant
                    mean_scores[l,a]  = self.predict_effect((mut_motif, position))

        return mean_scores



    def positional_bias(self, motif='UGCAUG', positions=[2, 12, 23, 33]):

        # loop over positions and measure effect size of intervention
        all_scores = []
        for position in positions:
            all_scores.append(self.predict_effect((motif, position)))

        return np.array(all_scores)



    def multiple_sites(self, motif='UGCAUG', positions=[17, 10, 25, 3]):

        # loop over positions and measure effect size of intervention
        all_scores = []
        for i, position in enumerate(positions):

            # embed motif multiple times
            interventions = []
            for j in range(i+1):
                interventions.append((motif_index, positions[j]))

            all_scores.append(self.predict_effect(interventions))

        return np.array(all_scores)


    def gc_bias(self, motif='UGCAUG', motif_position=17,
                gc_motif='GCGCGC', gc_positions=[34, 2]):

        all_scores = []


        # background sequence with gc-bias on right side
        all_scores.append(self.predict_effect((gc_motif_index, gc_positions[0])))

        # background sequence with motif at center
        all_scores.append(self.predict_effect((motif_index, motif_position)))

        # create interventions for gc bias
        for position in gc_positions:

            interventions = [(motif_index, motif_position), (gc_motif_index, position)]
            all_scores.append(self.predict_effect(interventions))

        return np.array(all_scores)




def pearsonr_scores(y_true, y_pred, mask_value=None):
    corr = []
    for i in range(y_true.shape[1]):
        if mask_value:
            index = np.where(y_true[:,i] != mask_value)[0]
            corr.append(stats.pearsonr(y_true[index,i], y_pred[index,i])[0])
        else:
            corr.append(stats.pearsonr(y_true[:,i], y_pred[:,i])[0])
    return np.array(corr)


"""

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
    nn = keras.layers.AveragePooling1D(pool_size=3,  # before it was max pool and pool size of 5 = 0.7834
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
    outputs = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
        
    return inputs, outputs
"""
