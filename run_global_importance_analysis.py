import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from residualbind import ResidualBind, GlobalImportance
import helper

#---------------------------------------------------------------------

normalization = 'clip_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
data_path = '../data/RNAcompete_2013/rnacompete2013.h5'
results_path = os.path.join('..','results')
save_path = os.path.join(results_path, normalization+'_'+ss_type)
plot_path = helper.make_directory(save_path, 'plots')

#---------------------------------------------------------------------------------------

# loop over different RNA binding proteins
experiments = helper.get_experiment_names(data_path)
for rbp_index, experiment in enumerate(experiments):
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
    model.load_weights()

    # instantiate global importance
    gi = GlobalImportance(model, class_index=0, alphabet='ACGU')

    # set null sequence model
    null_seq_model = np.mean(np.squeeze(train['inputs']), axis=0)
    null_seq_model /= np.sum(null_seq_model, axis=1, keepdims=True)
    gi.set_null_model(null_seq_model, num_sim=1000)

    # k-mer analysis to find motif
    kmer_size = 6
    position = 17
    kmers, mean_scores = gi.optimal_kmer(kmer_size, position)

    # save top kmers to file
    with open(os.path.join(plot_path, experiment + '_kmer.txt'), 'w') as f:
        for i in range(10):
            f.write("%d\t%.3f"%(kmers[i], mean_scores[i]))

    # Multiple sites
    motif = kmers[0]
    mean_scores = gi.kmer_mutagenesis(motif, position=17)

    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(mean_scores.T - np.max(mean_scores), cmap='hot')
    plt.xlabel('Positions', fontsize=22)
    plt.xticks(range(len(motif)), range(1,len(motif)+1), fontsize=22, ha='center');
    plt.yticks([0,1,2,3], ['A', 'C', 'G', 'U'], fontsize=22)
    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.05, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, fraction=95, pad=0.1) 
    cbar.ax.tick_params(labelsize=22) 
    plt.ylabel('$\Delta$ P', fontsize=22);
    outfile = os.path.join(plot_path, experiment+'_mutagenesis.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # positional analysis
    positions = [2, 8, 14, 20, 26, 32]
    mean_scores, all_scores = gi.positional_bias(motif, positions)

    fig = plt.figure()
    plt.boxplot(all_scores[1:,:].T, showfliers=False);
    plt.xticks(range(1,len(positions)+1), positions, fontsize=14, ha='center');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Effect size', fontsize=14)
    plt.xlabel('Positions', fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_motif_location.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # Multiple sites
    positions = [23, 2, 12, 30]
    mean_scores, all_scores = gi.multiple_sites(motif, positions)

    fig = plt.figure()
    plt.boxplot(all_scores[1:,:].T, showfliers=False);
    plt.xticks(range(1,len(positions)+1), [motif+' (x1)', motif+' (x2)', motif+' (x3)', motif+' (x4)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=12)
    plt.ylabel('Effect size', fontsize=14);
    outfile = os.path.join(plot_path, experiment+'_multiple_sites.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # GC Bias 
    motif_position = 17
    gc_motif = 'GCGCGC'
    gc_positions = [34, 2]
    mean_scoress, all_scores = gi.gc_bias(motif, motif_position,
                                          gc_motif, gc_positions)

    fig = plt.figure()
    plt.boxplot(all_scores[:,:].T, showfliers=False);
    plt.xticks([1, 2, 3, 4, 5, 6], ['Random', 'GC (right)', 'Motif', 'Motif+GC (right)', 'Motif+GC (left)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Effect size', fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_gcbias.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')




"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow import keras
from keras import backend as K

import helper
import global_importance as gi

#---------------------------------------------------------------------

model_name = 'residualbind'
normalize_method = 'clip_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
alphabet = 'ACGU'
results_path = os.path.join('..','results')
save_path = os.path.join(results_path, model_name+'_'+normalize_method+'_'+ss_type)
plot_path = helper.make_directory(save_path, 'plots')

# load experiment names
data_path = '../data/RNAcompete_2013/rnacompete2013.h5'
experiments = helper.get_experiments_hdf5(data_path)

#---------------------------------------------------------------------

for rbp_index, experiment in enumerate(experiments):
    K.clear_session()
    experiment = experiment.decode('UTF-8')
    print('Analyzing: '+ experiment)

    # load rbp dataset
    train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

    # process rbp dataset
    train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

    # load model
    inputs, outputs = helper.load_model(model_name)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # set up optimizer and metrics
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    # set to best model
    weights_path = os.path.join(save_path, experiment + '_weights.hdf5')
    model.load_weights(weights_path)

    # generate null sequences
    profile = np.mean(np.squeeze(train['inputs']), axis=0)
    profile /= np.sum(profile, axis=1, keepdims=True)
    null_sequence = gi.sample_profile_sequences(profile, num_sim=1000)

    # k-mer analysis to find motif
    kmer_size = 6
    position = 17
    kmers, mean_scores = gi.optimal_kmer(model, null_sequence, kmer_size, position, alphabet)

    # save top kmers to file
    with open(os.path.join(plot_path, experiment + '_kmer.txt'), 'w') as f:
        for i in range(10):
            f.write("%d\t%.3f"%(kmers[i], mean_scores[i]))

    # Multiple sites
    motif = kmers[0]
    position = 17
    mean_scores = gi.mutagenesis(model, null_sequence, motif, position, alphabet)

    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(mean_scores.T - np.max(mean_scores), cmap='hot')
    plt.xlabel('Positions', fontsize=22)
    plt.xticks(range(len(motif)), range(1,len(motif)+1), fontsize=22, ha='center');
    plt.yticks([0,1,2,3], ['A', 'C', 'G', 'U'], fontsize=22)
    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.05, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, fraction=95, pad=0.1) 
    cbar.ax.tick_params(labelsize=22) 
    plt.ylabel('$\Delta$ P', fontsize=22);
    outfile = os.path.join(plot_path, experiment+'_mutagenesis.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # positional analysis
    positions = [2, 8, 14, 20, 26, 32]
    mean_scores, all_scores = gi.positional_bias(model, null_sequence, motif, positions, alphabet)

    fig = plt.figure()
    plt.boxplot(all_scores[1:,:].T, showfliers=False);
    plt.xticks(range(1,len(positions)+1), positions, fontsize=14, ha='center');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Global importance', fontsize=14)
    plt.xlabel('Positions', fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_motif_location.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # Multiple sites
    positions = [23, 2, 12, 30]
    mean_scores, all_scores = gi.multiple_sites(model, null_sequence, motif, positions, alphabet)

    fig = plt.figure()
    plt.boxplot(all_scores[1:,:].T, showfliers=False);
    plt.xticks(range(1,len(positions)+1), [motif+' (x1)', motif+' (x2)', motif+' (x3)', motif+' (x4)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=12)
    plt.ylabel('Global importance', fontsize=14);
    outfile = os.path.join(plot_path, experiment+'_multiple_sites.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    # GC Bias 
    motif_position = 17
    gc_motif = 'GCGCGC'
    gc_positions = [34, 2]
    mean_scoress, all_scores = gi.gc_bias(model, null_sequence, motif, motif_position,
                                          gc_motif, gc_positions, alphabet)

    fig = plt.figure()
    plt.boxplot(all_scores[:,:].T, showfliers=False);
    plt.xticks([1, 2, 3, 4, 5, 6], ['Random', 'GC (right)', 'Motif', 'Motif+GC (right)', 'Motif+GC (left)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Global importance', fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_gcbias.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
"""