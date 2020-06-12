import os, sys, h5py
import numpy as np
import itertools
from six.moves import cPickle
import helper



def convert_index_to_one_hot(sequence_index, alphabet='ACGU'):
    one_hot = np.zeros((len(sequence_index), len(sequence_index[0]), len(alphabet)))
    for i, seq_index in enumerate(sequence_index):
        for j, letter in enumerate(seq_index):
            one_hot[i,j,letter] = 1.0
    return one_hot


def generate_profile(data_path):
    train, valid, test = helper.load_dataset_hdf5(data_path, ss_type='seq', rbp_index=0)
    return np.mean(np.squeeze(train['inputs']), axis=0)


def sample_profile_sequences(profile, num_sim=1000):
   
    # sequence length
    L = profile.shape[0]

    one_hot_seq = np.zeros((num_sim, L, 4))
    for n in range(num_sim):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0,1,L)

        # calculate cumulative sum of the probabilities
        cum_prob = profile.cumsum(axis=1)

        # find bin that matches random number for each position
        for l in range(L):
            index=[j for j in range(4) if Z[l] < cum_prob[l,j]][0]
            one_hot_seq[n,l,index] = 1
            
    return one_hot_seq




def embed_patterns_index(patterns, null_sequence):

    if not isinstance(patterns, list):
        patterns = [patterns]

    new_sequence = np.copy(null_sequence)
    for pattern, position in patterns:
        new_sequence[:,position:position+len(pattern)] = pattern

    return new_sequence
              

def optimal_kmer(model, null_sequence, kmer_size=7, position=17, alphabet='ACGU', class_index=0):
    
    # get null scores
    null_scores = model.predict(null_sequence)[:,class_index]
    mean_null_score = np.mean(null_scores)

    # convert to sequence of indices
    null_sequence_index = np.argmax(null_sequence, axis=2)

    # generate all kmers             
    kmers = ["".join(p) for p in itertools.product(list(alphabet), repeat=kmer_size)]

    # score each kmer
    mean_scores = []
    for i, kmer in enumerate(kmers):
        if np.mod(i+1,500) == 0:
            print("%d out of %d"%(i+1, len(kmers)))
        
        # convert kmer to categorical representation
        kmer_index = np.array([alphabet.index(i) for i in kmer])
        
        # embed kmer in random seqeunce
        sequence_index = embed_patterns_index((kmer_index, position), null_sequence_index)

        # convert to one hot representation
        one_hot = convert_index_to_one_hot(sequence_index, alphabet)

        # get predictions
        predictions = model.predict(one_hot)[:,class_index]

        mean_scores.append(np.mean(predictions) - mean_null_score)

    kmers = np.array(kmers)
    mean_scores = np.array(mean_scores)

    # sort by highest prediction
    sort_index = np.argsort(mean_scores)[::-1]

    return kmers[sort_index], mean_scores[sort_index]


def mutagenesis(model, null_sequence, motif='UGCAUG', position=17, alphabet='ACGU', class_index=0):
    
    L = len(motif)
    A = len(alphabet)

    # get null scores
    null_scores = model.predict(null_sequence)[:,class_index]
    mean_null_score = np.mean(null_scores)

    # convert to sequence of indices
    null_sequence_index = np.argmax(null_sequence, axis=2)

    # get wt score
    motif_index = np.array([alphabet.index(i) for i in motif])
    sequence_index = embed_patterns_index((motif_index, position), null_sequence_index)
    one_hot = convert_index_to_one_hot(sequence_index, alphabet)
    predictions = model.predict(one_hot)[:,class_index]
    wt_score = np.mean(predictions) - mean_null_score

    # score each kmer
    mean_scores = np.zeros((L, A))
    for l in range(L):
        for a in range(A):
            if motif[l] == alphabet[a]:
                mean_scores[l,a] = wt_score
            else:
                mut_motif_index = np.copy(motif_index)
                mut_motif_index[l] = a
                
                # embed kmer in random seqeunce
                sequence_index = embed_patterns_index((mut_motif_index, position), null_sequence_index)

                # convert to one hot representation
                one_hot = convert_index_to_one_hot(sequence_index, alphabet)

                # get predictions
                predictions = model.predict(one_hot)[:,class_index]

                mean_scores[l,a] = np.mean(predictions) - mean_null_score

    return mean_scores


def positional_bias(model, null_sequence, motif='UGCAUG', positions=[2, 12, 23, 33], 
                    alphabet='ACGU', class_index=0):

    # convert motif to categorical representation
    motif_index = np.array([alphabet.index(i) for i in motif])

    # get null scores
    null_scores = model.predict(null_sequence)[:,class_index]
    mean_null_score = np.mean(null_scores)

    # convert sequence to categorical representation
    null_sequence_index = np.argmax(np.squeeze(null_sequence), axis=2)

    mean_scores = [mean_null_score]
    all_scores = [null_scores]
    for position in positions:

        # embed kmer in random seqeunce
        sequence_index = embed_patterns_index((motif_index, position), null_sequence_index)
    
        # convert to one hot representation
        one_hot = convert_index_to_one_hot(sequence_index, alphabet)

        # get predictions
        predictions = model.predict(one_hot)[:,class_index]

        mean_scores.append(np.mean(predictions) - mean_null_score)
        all_scores.append(predictions)

    mean_scores = np.array(mean_scores)
    all_scores = np.array(all_scores)

    return mean_scores, all_scores



def multiple_sites(model, null_sequence, motif='UGCAUG', positions=[17, 10, 25, 3], 
                   alphabet='ACGU', class_index=0):

    # convert motif to categorical representation
    motif_index = np.array([alphabet.index(i) for i in motif])

    # get null scores
    null_scores = model.predict(null_sequence)[:,class_index]
    mean_null_score = np.mean(null_scores)

    # convert sequence to categorical representation
    null_sequence_index = np.argmax(np.squeeze(null_sequence), axis=2)

    mean_scores = [mean_null_score]
    all_scores = [null_scores]
    for i, position in enumerate(positions):

        interventions = []
        for j in range(i+1):
            interventions.append((motif_index, positions[j]))

        # embed kmer in random seqeunce
        sequence_index = embed_patterns_index(interventions, null_sequence_index)
    
        # convert to one hot representation
        one_hot = convert_index_to_one_hot(sequence_index, alphabet)

        # get predictions
        predictions = model.predict(one_hot)[:,class_index]

        mean_scores.append(np.mean(predictions))
        all_scores.append(predictions)

    mean_scores = np.array(mean_scores)
    all_scores = np.array(all_scores)

    return mean_scores, all_scores 



def gc_bias(model, null_sequence, motif='UGCAUG', motif_position=17,
            gc_motif='GCGCGC', gc_positions=[34, 2], alphabet='ACGU', class_index=0):

    mean_scores = []
    all_scores = []

    # convert motif to categorical representation
    motif_index = np.array([alphabet.index(i) for i in motif])
    gc_motif_index = np.array([alphabet.index(i) for i in gc_motif])

    # get null scores
    null_scores = model.predict(null_sequence)[:,class_index]
    mean_null_score = np.mean(null_scores)
    mean_scores.append(mean_null_score)
    all_scores.append(null_scores)

    # convert sequence to categorical representation
    null_sequence_index = np.argmax(np.squeeze(null_sequence), axis=2)

    # background sequence with gc-bias on right side
    sequence_index = embed_patterns_index((gc_motif_index, gc_positions[0]), null_sequence_index)
    one_hot = convert_index_to_one_hot(sequence_index, alphabet)
    scores = model.predict(one_hot)[:,class_index]
    mean_scores.append(np.mean(scores))
    all_scores.append(scores)

    # background sequence with motif at center
    sequence_index = embed_patterns_index((motif_index, motif_position), null_sequence_index)
    one_hot = convert_index_to_one_hot(sequence_index, alphabet)
    scores = model.predict(one_hot)[:,class_index]
    mean_scores.append(np.mean(scores))
    all_scores.append(scores)

    # create interventions for gc bias
    for position in gc_positions:
        interventions = [(motif_index, motif_position), (gc_motif_index, position)]
        sequence_index = embed_patterns_index(interventions, null_sequence_index)
        one_hot = convert_index_to_one_hot(sequence_index, alphabet)
        scores = model.predict(one_hot)[:,class_index]
        mean_scores.append(np.mean(scores))
        all_scores.append(scores)

    mean_scores = np.array(mean_scores)
    all_scores = np.array(all_scores)

    return mean_scores, all_scores


