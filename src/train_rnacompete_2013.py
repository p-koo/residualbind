import os, h5py
from six.moves import cPickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import helper

"""
residualbind: 0.6669+/-0.1767
dilatedresidualbind: 0.6725+/-0.1721
dilated2residualbind: 0.6707+/-0.1693
dilated_exp: 0.6830+/-0.1703
dilated_relu: 0.6795+/-0.1715
dilated2_exp: 0.6832+/-0.1696
dilated2_relu: 0.6807+/-0.1706
dilated3residualbind_exp: 0.6882+/-0.1712
dilated3residualbind_exp: 0.6882+/-0.1712
dilated3_exp2: 0.6875+/-0.1711
"""

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

