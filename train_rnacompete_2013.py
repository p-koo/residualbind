import os
import numpy as np
from tensorflow.keras import backend as K
from residualbind import ResidualBind
import helper

#---------------------------------------------------------------------------------------

normalization = 'clip_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
data_path = '../data/RNAcompete_2013/rnacompete2013.h5'
results_path = helper.make_directory('../results', 'rnacompete_2013')
save_path = helper.make_directory(results_path, normalization+'_'+ss_type)

#---------------------------------------------------------------------------------------

# loop over different RNA binding proteins
pearsonr_scores = []
experiments = helper.get_experiment_names(data_path)
for rbp_index, experiment in enumerate(experiments[189:]):
    experiment = experiment.decode('UTF-8')
    print('Analyzing: '+ experiment)

    # load rbp dataset
    train, valid, test = helper.load_rnacompete_data(data_path, 
                                                     ss_type=ss_type, 
                                                     normalization=normalization, 
                                                     rbp_index=rbp_index)

    # load residualbind model
    input_shape = list(train['inputs'].shape)[1:]
    weights_path = os.path.join(save_path, experiment + '_weights.hdf5')    
    model = ResidualBind(input_shape, weights_path)

    # fit model
    model.fit(train, valid, num_epochs=300, batch_size=100, patience=25, 
              lr=0.001, lr_decay=0.3, decay_patience=7)
    
    # evaluate model
    corr = model.test_model(test, batch_size=100, weights='best')
    print("  Test: "+str(np.mean(corr)))

    pearsonr_scores.append(corr)
pearsonr_scores = np.array(pearsonr_scores)

print('FINAL RESULTS: %.4f+/-%.4f'%(np.mean(pearsonr_scores), np.std(pearsonr_scores)))

# save results to table
file_path = os.path.join(results_path, normalization+'_'+ss_type+'_performance.tsv')
f.write('%s\t%s\n'%('Experiment', 'Pearson score'))
with open(file_path, 'w') as f:
    for experiment, score in zip(experiments, pearsonr_scores):
        f.write('%s\t%.4f\n'%(experiment, score))


"""

import os, h5py
from six.moves import cPickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import helper


#---------------------------------------------------------------------------------------

# different deep learning models to try out
model_name = 'residualbind'
normalize_method = 'clip_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'

data_path = '../data/RNAcompete_2013/rnacompete2013.h5'
results_path = helper.make_directory('..','results')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

save_path = helper.make_directory(results_path, model_name+'_'+normalize_method+'_'+ss_type)

# loop over different RNA binding proteins
all_results = []
for rbp_index, experiment in enumerate(experiments):
    K.clear_session()

    experiment = experiment.decode('UTF-8')
    print('Analyzing: '+ experiment)

    # load rbp dataset
    train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

    # process rbp dataset
    train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

    x_train, y_train = train['inputs'], train['targets']
    x_valid, y_valid = valid['inputs'], valid['targets']
    x_test,  y_test  = test['inputs'],  test['targets']

    # load model
    inputs, outputs = helper.load_model(model_name)

    # Instantiate model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # set up optimizer and metrics
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
    model.summary()
    
    # fit model with callbacks
    num_epochs = 300
    batch_size = 100 
    patience = 25
    lr_decay = 0.3
    decay_patience = 7
    
    lr = 0.001
    best_pearsonr = 0
    counter = 0
    decay_counter = 0
    for epoch in range(num_epochs):
        print('Epoch %d out of %d'%(epoch, num_epochs))

        # training epoch
        history = model.fit(x_train, y_train, 
                            epochs=1,
                            batch_size=batch_size, 
                            shuffle=True)

        predictions = model.predict(x_valid, batch_size=512)
        corr = helper.pearsonr_scores(y_valid, predictions)
        print('  Validation: ' + str(np.mean(corr)))

        if best_pearsonr < corr:
            best_pearsonr = corr
            decay_counter = 0
            counter = 0
            weights_path = os.path.join(save_path, experiment + '_weights.hdf5')
            model.save_weights(weights_path)
            print('  Saving model to: ' + weights_path)
        else:
            counter += 1
            decay_counter += 1
            if decay_counter == decay_patience:
                lr *= lr_decay
                lr = np.maximum(lr, 1e-6)
                K.set_value(model.optimizer.lr, lr)
                decay_counter = 0
                print('  Decaying learning rate to: %f'%(lr))

            if counter == patience:
                print('  Patience ran out... Early Stopping!')
                break

    # set to best model
    model.load_weights(weights_path)

    # evaluate model
    predictions = model.predict(x_test, batch_size=512)
    corr = helper.pearsonr_scores(y_test, predictions)
    print("  Test: "+str(np.mean(corr)))

    all_results.append(corr)
all_results = np.array(all_results)

print('FINAL RESULTS: %.4f+/-%.4f'%(np.mean(all_results), np.std(all_results)))

# save results to pickle file
with open(os.path.join(results_path, model_name+'.pickle'), 'wb') as f:
    cPickle.dump(all_results, f)

"""
