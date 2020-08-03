import os
import numpy as np
from tensorflow.keras import backend as K
from residualbind import ResidualBind
import helper

#---------------------------------------------------------------------------------------


normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
data_path = '../data/RNAcompete_2009/rnacompete2009.h5'
results_path = helper.make_directory('../results_final2', 'rnacompete_2009')
save_path = helper.make_directory(results_path, normalization+'_'+ss_type)

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
rbp_names = ['VTS1']#'Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1']



# loop over different RNA binding proteins
pearsonr_scores = []
for rbp_name in rbp_names:
    print('Analyzing: '+ rbp_name)
    file_path = os.path.join(save_path, rbp_name)

    # load rbp dataset
    train, valid, test = helper.load_rnacompete_data(data_path, 
                                                     ss_type=ss_type, 
                                                     normalization=normalization, 
                                                     dataset_name=rbp_name)

    # load residualbind model
    input_shape = list(train['inputs'].shape)[1:]
    weights_path = os.path.join(save_path, rbp_name + '_weights.hdf5')    
    model = ResidualBind(input_shape, weights_path)

    # fit model
    model.fit(train, valid, num_epochs=300, batch_size=100, patience=20, 
              lr=0.001, lr_decay=0.3, decay_patience=7)
        
    # evaluate model
    corr = model.test_model(test, batch_size=100, weights='best')
    print("  Test: "+str(np.mean(corr)))

    pearsonr_scores.append(corr)
pearsonr_scores = np.array(pearsonr_scores)

print('FINAL RESULTS: %.4f+/-%.4f'%(np.mean(pearsonr_scores), np.std(pearsonr_scores)))

# save results to table
file_path = os.path.join(results_path, normalization+'_'+ss_type+'_performance.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\n'%('Experiment', 'Pearson score'))
    for rbp_name, score in zip(rbp_names, pearsonr_scores):
        f.write('%s\t%.4f\n'%(rbp_name, score))

