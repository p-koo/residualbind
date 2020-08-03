#---------------------------------------------------------------------------------------
"""
Summary: Generates hdf5 file with all 2013 RNAcompete experiments.
"""
#---------------------------------------------------------------------------------------

import os, sys, h5py
import pandas as pd
import numpy as np
from six.moves import cPickle
np.random.seed(100)

#---------------------------------------------------------------------------------------


def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""

    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


def generate_fasta(sequences, fasta_path):
    """generate fasta file from an array of sequences
    """

    with open(fasta_path, 'w+') as f:
        for i in range(len(sequences)):
            f.write('>seq '+str(i))
            f.write('\n')
            f.write(sequences[i])
            f.write('\n')


def predict_structure(fasta_path, profile_path, window):
    """predict secondary structure profiles with RNAplfold modified scripts"""

    E_path = profile_path+'E_profile.txt'
    # predict external loops
    os.system('E_RNAplfold -W '+str(window)+' -u 1 <'+fasta_path+' >'+E_path)

    # predict hairpin loops
    H_path = profile_path+'H_profile.txt'
    os.system('H_RNAplfold -W '+str(window)+' -u 1 <'+fasta_path+' >'+H_path)

    # predict internal loops
    I_path = profile_path+'I_profile.txt'
    os.system('I_RNAplfold -W '+str(window)+' -u 1 <'+fasta_path+' >'+I_path)

    # predict multi-loops
    M_path = profile_path+ 'M_profile.txt'
    os.system('M_RNAplfold -W '+str(window)+' -u 1 <'+fasta_path+' >'+M_path)



def merge_structural_profile(profile_path, merged_path):
    """merge the secondary structure profiles into a single file"""
    def list_to_str(lst):
        ''' Given a list, return the string of that list with tab separators
        '''
        return reduce( (lambda s, f: s + '\t' + str(f)), lst, '')

    # external loop profile
    E_path = profile_path+'E_profile.txt'
    fEprofile = open(E_path)
    Eprofiles = fEprofile.readlines()

    # hairpin loop profiles
    H_path = profile_path+'H_profile.txt'
    fHprofile = open(H_path)
    Hprofiles = fHprofile.readlines()

    # internal loop profiles
    I_path = profile_path+'I_profile.txt'
    fIprofile = open(I_path)
    Iprofiles = fIprofile.readlines()

    # multi-loop profiles
    M_path = profile_path+ 'M_profile.txt'
    fMprofile = open(M_path)
    Mprofiles = fMprofile.readlines()

    num_seq = int(len(Eprofiles)/2)

    # parse into a single file
    fhout = open(merged_path, 'w')
    for i in range(num_seq):
        id = Eprofiles[i*2].split()[0]
        fhout.write(id+'\n')
        H_prob =  Hprofiles[i*2+1].split()
        I_prob =  Iprofiles[i*2+1].split()
        M_prob =  Mprofiles[i*2+1].split()
        E_prob =  Eprofiles[i*2+1].split()
        P_prob = map( (lambda a, b, c, d: 1-float(a)-float(b)-float(c)-float(d)), H_prob, I_prob, M_prob, E_prob)
        fhout.write(list_to_str(P_prob[:len(P_prob)])+'\n')
        fhout.write(list_to_str(H_prob[:len(P_prob)])+'\n')
        fhout.write(list_to_str(I_prob[:len(P_prob)])+'\n')
        fhout.write(list_to_str(M_prob[:len(P_prob)])+'\n')
        fhout.write(list_to_str(E_prob[:len(P_prob)])+'\n')
    fhout.close()

    return num_seq


def extract_structural_profile(merged_path, num_seq, window):
    """extract secondary structure profiles from a merged file and return a
       numpy array """

    # parse further and load structural profile as np.array
    f = open(merged_path, 'r')
    structure = []
    for i in range(num_seq):
        seq = f.readline()
        paired = f.readline().strip().split('\t')
        hairpin = f.readline().strip().split('\t')
        internal = f.readline().strip().split('\t')
        multi = f.readline().strip().split('\t')
        external = f.readline().strip().split('\t')

        paired = np.array(paired).astype(np.float32)
        hairpin = np.array(hairpin).astype(np.float32)
        internal = np.array(internal).astype(np.float32)
        multi = np.array(multi).astype(np.float32)
        external = np.array(external).astype(np.float32)

        # pad sequences
        seq_length = len(paired)
        offset1 = int((window - seq_length)/2)
        offset2 = window - seq_length - offset1
        struct = np.array([paired, hairpin, internal, multi, external])
        num_dims = struct.shape[0]
        if offset1:
            struct = np.hstack([np.zeros((num_dims,offset1)), struct])
        if offset2:
            struct = np.hstack([struct, np.zeros((num_dims,offset2))])
        structure.append(struct)

    return np.array(structure)


#---------------------------------------------------------------------------------------


data_path = '../../data/RNAcompete_2013'

# load binding affinities for each rnacompete experiment
df = pd.read_csv(os.path.join(data_path,'targets.tsv'), sep='\t')
targets = df.to_numpy()
experiments = [x.encode('UTF8') for x in df.columns.values]

# load sequences
df = pd.read_csv(os.path.join(data_path,'sequences.tsv'), sep='\t')
rnac_set = df['Fold ID'].to_numpy()
sequences = df['seq'].to_numpy()

# get the maximum length sequence
max_length = 0
for seq in sequences:
    max_length = np.maximum(max_length, len(seq))

# convert sequences into one-hot representation
one_hot = convert_one_hot(sequences, max_length)

# save sequences in a fasta format (for rnaplfold)
fasta_path = os.path.join(data_path,'sequences.fa')
generate_fasta(sequences, fasta_path)

# generate secondary structure profiles with rnaplfold
profile_path = os.path.join(data_path,'rnaplfold')
predict_structure(fasta_path, profile_path, window=max_length)

# generate merged secondary structure profile
merged_path = profile_path+'_structure_profiles.txt'
num_seq = merge_structural_profile(profile_path, merged_path)

# extract secondary structure profiles
structure = extract_structural_profile(merged_path, num_seq, window)

# merge sequences and structural profiles
data = np.concatenate([one_hot, structure], axis=1)
data = one_hot

# split dataset into train, cross-validation, and test set
valid_frac = 0.1
index = np.where(rnac_set == 'A')[0]
num_seq = len(index)
num_valid = int(num_seq*valid_frac)
shuffle = np.random.permutation(num_seq)

X_train = data[shuffle[num_valid:]]
Y_train = targets[[shuffle[num_valid:]]]

X_valid = data[shuffle[:num_valid]]
Y_valid = targets[[shuffle[:num_valid]]]

test_index = np.where(rnac_set == 'B')[0]
X_test = data[test_index]
Y_test = targets[test_index]

# save dataset
save_path = os.path.join(data_path, 'rnacompete2013.h5')
print('saving dataset: ', save_path)
with h5py.File(save_path, "w") as f:
    dset = f.create_dataset("X_train", data=X_train.astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_train", data=Y_train.astype(np.float32), compression="gzip")
    dset = f.create_dataset("X_valid", data=X_valid.astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_valid", data=Y_valid.astype(np.float32), compression="gzip")
    dset = f.create_dataset("X_test", data=X_test.astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_test", data=Y_test.astype(np.float32), compression="gzip")
    dset = f.create_dataset("experiment", data=experiments, compression="gzip")
