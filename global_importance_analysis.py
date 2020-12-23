import os
import pandas as pd
import numpy as np
import logomaker
from six.moves import cPickle
import matplotlib.pyplot as plt
from scipy import stats
from residualbind import ResidualBind, GlobalImportance
import helper, explain

#---------------------------------------------------------------------

normalization = 'log_norm'   # 'log_norm' or 'clip_norm'
ss_type = 'seq'                  # 'seq', 'pu', or 'struct'
data_path = '../data/RNAcompete_2013/rnacompete2013.h5'
results_path = os.path.join('../results', 'rnacompete_2013')
save_path = os.path.join(results_path, normalization+'_'+ss_type)
plot_path = helper.make_directory(save_path, 'plots')
motif_path = helper.make_directory(save_path, 'motifs')
kmer_path = helper.make_directory(save_path, 'kmer_motifs')
alphabet = 'ACGU'

#---------------------------------------------------------------------------------------

# get experiment names
experiments = helper.get_experiment_names(data_path)

# loop over different RNA binding proteins
multiple_sites_all = []
gcbias_all = []
hairpin_all = []
for rbp_index, experiment in enumerate(experiments):
    print(rbp_index, experiment)

    # load rbp dataset
    train, valid, test = helper.load_rnacompete_data(data_path, 
                                                     ss_type=ss_type, 
                                                     normalization=normalization, 
                                                     rbp_index=rbp_index)

    # load residualbind model
    input_shape = list(train['inputs'].shape)[1:]
    num_class = 1
    weights_path = os.path.join(save_path, experiment + '_weights.hdf5')    
    resnet = ResidualBind(input_shape, num_class, weights_path)
    resnet.load_weights()

    # instantiate global importance
    gi = GlobalImportance(resnet, alphabet)

    # set null sequence model
    null_seq_model = np.mean(np.squeeze(train['inputs']), axis=0)
    null_seq_model /= np.sum(null_seq_model, axis=1, keepdims=True)
    gi.set_null_model(null_seq_model, num_sim=1000)

    #-----------------------------------------------------------------------------
    # k-mer analysis to find motif
    print("performing k-mer analysis")

    kmer_size = 6
    position = 17
    kmers, mean_scores = gi.optimal_kmer(kmer_size, position, class_index=0)

    # save top kmers to file
    with open(os.path.join(plot_path, experiment + '_kmer.txt'), 'w') as f:
        for i in range(10):
            f.write("%s\t%.3f\n"%(kmers[i], mean_scores[i]))

    # set kmer to investigate
    motif = kmers[0]
       
    # generate motif from weighted k-mer alignment
    kmer_motif = explain.kmer_alignment_motif(kmers, mean_scores, alphabet)

    # convert to logo
    I = np.log2(4) + np.sum(kmer_motif * np.log2(kmer_motif+1e-10), axis=1, keepdims=True)
    logo = np.maximum(I*kmer_motif, 1e-7)

    # setup dataframe for logmaker       
    L = len(kmer_motif)
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
    for l in range(L):
        for a in range(4):
            counts_df.iloc[l,a] = logo[l,a]

    fig = plt.figure(figsize=(3,2))
    ax = plt.subplot(111)
    logomaker.Logo(counts_df, ax=ax)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])
    outfile = os.path.join(motif_path, experiment+'_kmer_motif_logo.png')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    #-----------------------------------------------------------------------------
    # kmer mutagenesis
    print("performing k-mer mutagenesis analysis")
    
    position = 17
    mean_scores = gi.kmer_mutagenesis(motif, position, class_index=0)

    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(mean_scores.T - np.max(mean_scores), cmap='hot')
    plt.xlabel('Positions', fontsize=22)
    plt.xticks(range(len(motif)), range(1,len(motif)+1), fontsize=22, ha='center');
    plt.yticks([0,1,2,3], ['A', 'C', 'G', 'U'], fontsize=22)
    plt.ylim([-0.5,3.5])
    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.05, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, fraction=95, pad=0.1) 
    cbar.ax.tick_params(labelsize=22) 
    plt.ylabel('$\Delta$ P', fontsize=22);
    outfile = os.path.join(plot_path, experiment+'_kmer_mutagenesis.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

    # generate a logo based on the sensitivity of each position in the highest scoring kmer
    kmer_motif = np.sqrt(np.sum(mean_scores**2, axis=1))

    # setup dataframe for logmaker       
    L = len(kmer_motif)
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
    for l in range(L):
        for a in range(4):
            if motif[l] == alphabet[a]:
                counts_df.iloc[l,a] = kmer_motif[l]
            else:
                counts_df.iloc[l,a] = 0

    fig = plt.figure(figsize=(3,2))
    ax = plt.subplot(111)
    logomaker.Logo(counts_df, ax=ax)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])
    outfile = os.path.join(kmer_path, experiment+'_kmer_logo.png')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    #-----------------------------------------------------------------------------
    # positional bias analysis
    print("performing positional bias analysis")

    positions = [3, 10, 16, 22, 28, 34]
    all_scores = gi.positional_bias(motif, positions, class_index=0)

    fig = plt.figure()
    flierprops = dict(marker='^', markerfacecolor='green', markersize=14, linestyle='none')
    plt.boxplot(all_scores.T, showfliers=False, showmeans=True, meanprops=flierprops);
    plt.xticks(range(1,len(positions)+1), positions, fontsize=14, ha='center');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Importance', fontsize=14)
    plt.xlabel('Positions', fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_position_bias.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')


    #-----------------------------------------------------------------------------
    # Multiple sites
    print("performing multiple sites analysis")

    positions = [4, 12, 20]
    all_scores = gi.multiple_sites(motif, positions, class_index=0)

    fig = plt.figure(figsize=(4,5))
    flierprops = dict(marker='^', markerfacecolor='green', markersize=14,linestyle='none')
    box = plt.boxplot(all_scores.T, showfliers=False, showmeans=True, meanprops=flierprops);
    plt.xticks(range(1,len(positions)+1), [motif+' (x1)', motif+' (x2)', motif+' (x3)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=12)
    plt.ylabel('Importance', fontsize=14);
    x = np.linspace(1,3,3)
    p = np.polyfit(x, np.mean(all_scores, axis=1), 1)
    determination = np.corrcoef(x, np.mean(all_scores, axis=1))[0,1]**2
    x = np.linspace(0.5,3.5,10)
    plt.plot(x, x*p[0] + p[1], '--k', alpha=0.5)
    MAX = 0
    for w in box['whiskers']:
        MAX = np.maximum(MAX, np.max(w.get_ydata()))
    scale = (np.percentile(all_scores, 90) - np.percentile(all_scores,10))/10
    plt.text(0.6, MAX-scale, "$R^2$ = %.3f"%(determination), fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_multiple_sites.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

    multiple_sites_all.append([np.mean(all_scores,axis=1),determination])


    #-----------------------------------------------------------------------------
    # GC Bias 
    print("performing gc-bias analysis")

    motif_position = 17
    gc_motif = 'GCGCGC'
    gc_positions = [34, 2]
    all_scores = gi.gc_bias(motif, motif_position, gc_motif, gc_positions, class_index=0)

    fig = plt.figure(figsize=(4,5))
    flierprops = dict(marker='^', markerfacecolor='green', markersize=14, linestyle='none')
    box = plt.boxplot(all_scores.T, showfliers=False, showmeans=True, meanprops=flierprops);
    plt.xticks([1, 2, 3, 4, 5], ['GC (right)', 'Motif', 'Motif+GC (right)', 'Motif+GC (left)'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Importance', fontsize=14)

    # hypothesis test whether means are similar
    MAX = 0
    for w in box['whiskers']:
        MAX = np.maximum(MAX, np.max(w.get_ydata()))
    scale = (np.percentile(all_scores, 90) - np.percentile(all_scores,10))/10
    pvalue1 = helper.add_significance(all_scores, start=2, end=3, height=MAX+scale, percentile=97.5, significance=0.5, fontsize=14)
    pvalue2 = helper.add_significance(all_scores, start=2, end=4, height=MAX+scale*3, percentile=97.5, significance=0.5, fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_gc_bias.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

    gcbias_all.append([np.mean(all_scores, axis=1), pvalue1, pvalue2])


    #-----------------------------------------------------------------------------
    # hairpin structure bias analysis
    print("performing hairpin bias analysis")

    positions = [17, 9]

    # embed patterns in same positions in random sequences
    all_scores = []
    for position in positions:
        all_scores.append(gi.embed_predict_effect((motif, position), class_index=0))

    # embed motif in loop and stem of hairpin sequence
    for position in positions:
        one_hot = gi.embed_pattern_hairpin((motif, position), stem_left=8, stem_right=24, stem_size=9)
        all_scores.append(gi.predict_effect(one_hot))
    all_scores = np.array(all_scores)
        
    fig = plt.figure(figsize=(4,5))
    flierprops = dict(marker='^', markerfacecolor='green', markersize=14, linestyle='none')
    box = plt.boxplot(all_scores.T, showfliers=False, showmeans=True, meanprops=flierprops);
    plt.xticks([1, 2, 3, 4], [ 'Motif-center', 'Motif-left', 'Motif-loop', 'Motif-stem'], rotation=40, fontsize=14, ha='right');
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(),fontsize=14);
    plt.ylabel('Importance', fontsize=14)

    # hypothesis test whether means are similar
    MAX = 0
    for w in box['whiskers']:
        MAX = np.maximum(MAX, np.max(w.get_ydata()))
    MAX
    scale = (np.percentile(all_scores, 90) - np.percentile(all_scores,10))/10
    pvalue1 = helper.add_significance(all_scores, start=1, end=3, height=MAX+scale, percentile=97.5, significance=0.5, fontsize=14)
    pvalue2 = helper.add_significance(all_scores, start=3, end=4, height=MAX+scale, percentile=97.5, significance=0.5, fontsize=14)
    pvalue3 = helper.add_significance(all_scores, start=2, end=4, height=MAX+scale*3, percentile=97.5, significance=0.5, fontsize=14)
    outfile = os.path.join(plot_path, experiment+'_hairpin_bias.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

    hairpin_all.append([np.mean(all_scores, axis=1), pvalue1, pvalue2, pvalue3])

# save main results
with open(os.path.join(plot_path, 'results.pickle'), 'wb') as f:
    cPickle.dump(np.array(multiple_sites_all), f)
    cPickle.dump(np.array(gcbias_all), f)
    cPickle.dump(np.array(hairpin_all), f)



