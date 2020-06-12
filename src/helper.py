import os, h5py
import numpy as np
from scipy import stats



def load_dataset_hdf5(file_path, dataset_name=None, ss_type='seq', rbp_index=None):

	def prepare_data(train, ss_type=None):

		seq = train['inputs'][:,:,:4]

		if ss_type == 'pu':
			structure = train['inputs'][:,:,4:9]
			paired = np.expand_dims(structure[:,:,0], axis=3)
			unpaired = np.expand_dims(np.sum(structure[:,:,1:], axis=3), axis=3)
			seq = np.concatenate([seq, paired, unpaired], axis=3)

		elif ss_type == 'struct':
			structure = train['inputs'][:,:,4:9]
			paired = np.expand_dims(structure[:,:,0], axis=3)
			HIME = structure[:,:,1:]
			seq = np.concatenate([seq, paired, HIME], axis=3)

		train['inputs']  = seq
		return train

	# open dataset
	experiments = None
	dataset = h5py.File(file_path, 'r')
	if not dataset_name:
		# load set A data
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
		X_train = np.array(dataset['/'+dataset_name+'/X_train']).astype(np.float32)
		Y_train = np.array(dataset['/'+dataset_name+'/Y_train']).astype(np.float32)
		X_valid = np.array(dataset['/'+dataset_name+'/X_valid']).astype(np.float32)
		Y_valid = np.array(dataset['/'+dataset_name+'/Y_valid']).astype(np.float32)
		X_test = np.array(dataset['/'+dataset_name+'/X_test']).astype(np.float32)
		Y_test = np.array(dataset['/'+dataset_name+'/Y_test']).astype(np.float32)

	# expand dims of targets
	if len(Y_train.shape) == 1:
		Y_train = np.expand_dims(Y_train, axis=1)
		Y_valid = np.expand_dims(Y_valid, axis=1)
		Y_test = np.expand_dims(Y_test, axis=1)

	# add another dimension to make a 4d tensor
	X_train = X_train.transpose([0, 2, 1])
	X_test = X_test.transpose([0, 2, 1])
	X_valid = X_valid.transpose([0, 2, 1])

	# dictionary for each dataset
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	# parse secondary structure profiles
	train = prepare_data(train, ss_type)
	valid = prepare_data(valid, ss_type)
	test = prepare_data(test, ss_type)

	return train, valid, test


def process_data(train, valid, test, method='log_norm'):
	"""get the results for a single experiment specified by rbp_index.
	Then, preprocess the binding affinity intensities according to method.
	method:
		clip_norm - clip datapoints larger than 4 standard deviations from the mean
		log_norm - log transcormation
		both - perform clip and log normalization as separate targets (expands dimensions of targets)
	"""

	def normalize_data(data, method):
		if method == 'standard':
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			sigma = np.mean(data)
			data_norm = (data)/sigma
			params = sigma
		if method == 'clip_norm':
			# standard-normal transformation
			significance = 4
			std = np.std(data)
			index = np.where(data > std*significance)[0]
			data[index] = std*significance
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [mu, sigma]

		elif method == 'log_norm':
			# log-standard-normal transformation
			MIN = np.min(data)
			data = np.log(data-MIN+1)
			mu = np.mean(data)
			sigma = np.std(data)
			data_norm = (data-mu)/sigma
			params = [MIN, mu, sigma]

		elif method == 'both':
			data_norm1, params = normalize_data(data, 'clip_norm')
			data_norm2, params = normalize_data(data, 'log_norm')
			data_norm = np.hstack([data_norm1, data_norm2])
		return data_norm, params


	# get binding affinities for a given rbp experiment
	Y_train = train['targets']
	Y_valid = valid['targets']
	Y_test = test['targets']

	# filter NaN
	train_index = np.where(np.isnan(Y_train) == False)[0]
	valid_index = np.where(np.isnan(Y_valid) == False)[0]
	test_index = np.where(np.isnan(Y_test) == False)[0]
	Y_train = Y_train[train_index]
	Y_valid = Y_valid[valid_index]
	Y_test = Y_test[test_index]
	X_train = train['inputs'][train_index]
	X_valid = valid['inputs'][valid_index]
	X_test = test['inputs'][test_index]

	# normalize intenensities
	if method:
		Y_train, params_train = normalize_data(Y_train, method)
		Y_valid, params_valid = normalize_data(Y_valid, method)
		Y_test, params_test = normalize_data(Y_test, method)

	# store sequences and intensities
	train = {'inputs': X_train, 'targets': Y_train}
	valid = {'inputs': X_valid, 'targets': Y_valid}
	test = {'inputs': X_test, 'targets': Y_test}

	return train, valid, test

def dataset_keys_hdf5(file_path):

	dataset = h5py.File(file_path, 'r')
	keys = []
	for key in dataset.keys():
		keys.append(str(key))

	return np.array(keys)


def get_experiments_hdf5(file_path):
	dataset = h5py.File(file_path, 'r')
	return np.array(dataset['experiment'])




def load_all_data(data_path, ss_type='seq', log_norm=True, mask_value=-10):

    def process_inputs(X_train, method='seq'):
        if ss_type == 'seq':
            data = X_train[:,:4]
        elif ss_type == 'pu':
            structure = X_train[:,4:9]
            paired = np.expand_dims(structure[:,0], axis=2)
            unpaired = np.expand_dims(np.sum(structure[:,1:], axis=2), axis=2)
            data = np.concatenate([X_train[:,:,:4], paired, unpaired], axis=2)
        elif ss_type == 'struct':
            data = X_train
        return data

    def log_normalization(Y_train):
        MIN = np.nanmin(Y_train, axis=1, keepdims=True)
        data = np.log(Y_train-MIN+1)
        mu = np.nanmean(data, axis=1, keepdims=True)
        sigma = np.nanstd(data, axis=1, keepdims=True)
        return (data-mu)/sigma

    def mask_nans(Y_train, value):
        nan_index = np.where(np.isnan(Y_train)==True)
        Y_train[nan_index[0], nan_index[1]] = value
        return Y_train

    # load dataset
    with h5py.File(data_path, 'r') as dataset:
        X_train = np.array(dataset['X_train']).astype(np.float32).transpose([0,2,1])
        Y_train = np.array(dataset['Y_train']).astype(np.float32)
        X_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0,2,1])
        Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
        X_test = np.array(dataset['X_test']).astype(np.float32).transpose([0,2,1])
        Y_test = np.array(dataset['Y_test']).astype(np.float32)
    N, L, A = X_train.shape

    # get sequence + secondary structure data
    X_train = process_inputs(X_train)
    X_valid = process_inputs(X_valid)
    X_test = process_inputs(X_test)

    # normalize targets
    Y_train = log_normalization(Y_train)
    Y_valid = log_normalization(Y_valid)
    Y_test = log_normalization(Y_test)

    if mask_value:
        Y_train = mask_nans(Y_train, mask_value)
        Y_valid = mask_nans(Y_valid, mask_value)
        Y_test = mask_nans(Y_test, mask_value)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def parse_experiment(x_train_all, y_train_all,
                     x_valid_all, y_valid_all,
                     x_test_all, y_test_all, class_index, mask_value=-10):
    index = np.where(np.not_equal(y_train_all[:,0], mask_value))[0]
    x_train = x_train_all[index]
    y_train = np.expand_dims(y_train_all[index, class_index], axis=1)

    index = np.where(np.not_equal(y_valid_all[:,0], mask_value))[0]
    x_valid = x_valid_all[index]
    y_valid = np.expand_dims(y_valid_all[index, class_index], axis=1)

    index = np.where(np.not_equal(y_test_all[:,0], mask_value))[0]
    x_test = x_test_all[index]
    y_test = np.expand_dims(y_test_all[index, class_index], axis=1)
    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_model(model_name):
    if model_name == 'residualbind':
        import residualbind
        inputs, outputs = residualbind.model()
    elif model_name == 'residualbind2':
        import residualbind2
        inputs, outputs = residualbind2.model()

    return inputs, outputs




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



def pearsonr_scores(y_true, y_pred, mask_value=None):
    corr = []
    for i in range(y_true.shape[1]):
        if mask_value:
            index = np.where(y_true[:,i] != mask_value)[0]
            corr.append(stats.pearsonr(y_true[index,i], y_pred[index,i])[0])
        else:
            corr.append(stats.pearsonr(y_true[:,i], y_pred[:,i])[0])
    return np.array(corr)

