import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1



def saliency(model, X, class_index=0, layer=-1, batch_size=256):
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]
    sess = K1.get_session()

    N = len(X)
    num_batches = int(np.floor(N/batch_size))

    attr_score = []
    for i in range(num_batches):
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[i*batch_size:(i+1)*batch_size]}))
    if num_batches*batch_size < N:
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[num_batches*batch_size:N]}))

    return np.concatenate(attr_score, axis=0)


def mutagenesis(model, X, class_index=0, layer=-1):

    def generate_mutagenesis(X):
        L,A = X.shape 

        X_mut = []
        for l in range(L):
            for a in range(A):
                X_new = np.copy(X)
                X_new[l,:] = 0
                X_new[l,a] = 1
                X_mut.append(X_new)
        return np.array(X_mut)

    N, L, A = X.shape 
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

    attr_score = []
    for x in X:

        # get baseline wildtype score
        wt_score = intermediate.predict(np.expand_dims(x, axis=0))[:, class_index]

        # generate mutagenized sequences
        x_mut = generate_mutagenesis(x)
        
        # get predictions of mutagenized sequences
        predictions = intermediate.predict(x_mut)[:,class_index]

        # reshape mutagenesis predictiosn
        mut_score = np.zeros((L,A))
        k = 0
        for l in range(L):
            for a in range(A):
                mut_score[l,a] = predictions[k]
                k += 1
                
        attr_score.append(mut_score - wt_score)
    return np.array(attr_score)



def kmer_alignment_motif(kmers, scores, alphabet='ACGU'):
    one_hot = np.zeros((len(kmers), len(kmers[0]), len(alphabet)))
    for n, kmer in enumerate(kmers):
        for l, a in enumerate(kmer):
            one_hot[n,l,alphabet.index(a)] = 1

    # zero pad highest scoring k-mer 
    N, L, A = one_hot.shape
    M = L*3
    base = np.concatenate([np.zeros((L,A)), one_hot[0], np.zeros((L,A))], axis=0)

    # make sure no negative weights
    scores = np.array(scores)

    if np.min(scores) < 0:
        scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores)) + 0.01

    # align other k-mers to highest scoring k-mer (weighted by GIA score)
    alignment = np.zeros((N,M,A))
    alignment[0,:,:] = base[0] * scores[0]
    for n in range(1,N):
        val = []
        for l in range(M-L):
            index = range(l, l+L)
            val.append(np.sum(base[index,:]*one_hot[n]))
        pos = np.argmax(val)
        alignment[n,:,:] = scores[n]*np.concatenate([np.zeros((pos,A)), one_hot[n], np.zeros((M-L-pos,A))], axis=0)

    # normalize alignment
    alignment = np.sum(alignment, axis=0)/np.sum(scores)

    # truncate positions with zeros
    index = np.where(np.sum(alignment, axis=1) != 0)[0]
    motif = alignment[index[0]:index[-1],:]
    return motif
