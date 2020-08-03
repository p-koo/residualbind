import os, h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir


def load_rnacompete_data(file_path, ss_type='seq', normalization='log_norm', rbp_index=None, dataset_name=None):

	def prepare_data(train, ss_type=None):

		seq = train['inputs'][:,:,:4]

		if ss_type == 'pu':
			structure = train['inputs'][:,:,4:9]
			paired = np.expand_dims(structure[:,:,0], axis=2)
			unpaired = np.expand_dims(np.sum(structure[:,:,1:], axis=2), axis=2)
			seq = np.concatenate([seq, paired, unpaired], axis=2)

		elif ss_type == 'struct':
			structure = train['inputs'][:,:,4:9]
			paired = np.expand_dims(structure[:,:,0], axis=2)
			HIME = structure[:,:,1:]
			seq = np.concatenate([seq, paired, HIME], axis=2)

		train['inputs']  = seq
		return train

	def normalize_data(data, normalization):
		if normalization == 'clip_norm':
			# standard-normal transformation
			significance = 4
			std = np.std(data)
			index = np.where(data > std*significance)[0]
			data[index] = std*significance
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [mu, sigma]

		elif normalization == 'log_norm':
			# log-standard-normal transformation
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [MIN, mu, sigma]
		return data_norm, params

	# open dataset
	dataset = h5py.File(file_path, 'r')
	if not dataset_name:  
		# load data from RNAcompete 2013
		X_train = np.array(dataset['X_train']).astype(np.float32)
		Y_train = np.array(dataset['Y_train']).astype(np.float32)
		X_valid = np.array(dataset['X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
		X_test = np.array(dataset['X_test']).astype(np.float32)
		Y_test = np.array(dataset['Y_test']).astype(np.float32)

		# expand dims of targets
		if rbp_index is not None:
			Y_train = Y_train[:,rbp_index]
			Y_valid = Y_valid[:,rbp_index]
			Y_test = Y_test[:,rbp_index]
	else:
		# necessary for RNAcompete 2009 dataset
		X_train = np.array(dataset['/'+dataset_name+'/X_train']).astype(np.float32)
		Y_train = np.array(dataset['/'+dataset_name+'/Y_train']).astype(np.float32)
		X_valid = np.array(dataset['/'+dataset_name+'/X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['/'+dataset_name+'/Y_valid']).astype(np.float32)
		X_test = np.array(dataset['/'+dataset_name+'/X_test']).astype(np.float32)
		Y_test = np.array(dataset['/'+dataset_name+'/Y_test']).astype(np.float32)

	# expand dims of targets if needed
	if len(Y_train.shape) == 1:
		Y_train = np.expand_dims(Y_train, axis=1)
		Y_valid = np.expand_dims(Y_valid, axis=1)
		Y_test = np.expand_dims(Y_test, axis=1)

	# transpose to make (N, L, A)
	X_train = X_train.transpose([0, 2, 1])
	X_test = X_test.transpose([0, 2, 1])
	X_valid = X_valid.transpose([0, 2, 1])

	# filter NaN
	train_index = np.where(np.isnan(Y_train) == False)[0]
	valid_index = np.where(np.isnan(Y_valid) == False)[0]
	test_index = np.where(np.isnan(Y_test) == False)[0]
	Y_train = Y_train[train_index]
	Y_valid = Y_valid[valid_index]
	Y_test = Y_test[test_index]
	X_train = X_train[train_index]
	X_valid = X_valid[valid_index]
	X_test = X_test[test_index]

	# normalize intenensities
	Y_train, params_train = normalize_data(Y_train, normalization)
	Y_valid, params_valid = normalize_data(Y_valid, normalization)
	Y_test, params_test = normalize_data(Y_test, normalization)

	# dictionary for each dataset
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	# parse secondary structure profiles
	train = prepare_data(train, ss_type)
	valid = prepare_data(valid, ss_type)
	test = prepare_data(test, ss_type)

	return train, valid, test



def dataset_keys_hdf5(file_path):
	dataset = h5py.File(file_path, 'r')
	keys = []
	for key in dataset.keys():
		keys.append(str(key))
	return np.array(keys)


def get_experiment_names(file_path):
	dataset = h5py.File(file_path, 'r')
	return [i.decode('UTF-8') for i in np.array(dataset['experiment'])]

def find_experiment_index(data_path, experiment):
    experiments = get_experiment_names(data_path)
    return experiments.index(experiment)

    
def add_significance(all_scores, start, end, height, percentile=97.5, significance=0.5, fontsize=14):

    def significance_bar(start,end,height,displaystring,linewidth = 1.2,
                         markersize = 8,boxpad=0.3,fontsize = 15,color = 'k'):
        from matplotlib.markers import TICKDOWN
        # draw a line with downticks at the ends
        plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
        # draw the text with a bounding box covering up the line
        plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

    #results = stats.wilcoxon(all_scores[start-1], all_scores[end-1])
    results = stats.ttest_ind(all_scores[start-1], all_scores[end-1], equal_var=False) # Welch’s t-test because unequal variance
    if results.pvalue > significance:
        displaystring = 'NS'
    else:
        displaystring = '*'
    significance_bar(start,end,height,displaystring,linewidth=1.2,markersize=8,
                     boxpad=0.3,fontsize=fontsize,color='k')
    return results.pvalue
