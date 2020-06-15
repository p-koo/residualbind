==========
neuralbinder
==========

::
        
	                             _ _     _           _           
	                            | | |   (_)         | |          
	  _ __   ___ _   _ _ __ __ _| | |__  _ _ __   __| | ___ _ __ 
	 | '_ \ / _ \ | | | '__/ _` | | '_ \| | '_ \ / _` |/ _ \ '__|
	 | | | |  __/ |_| | | | (_| | | |_) | | | | | (_| |  __/ |   
	 |_| |_|\___|\__,_|_|  \__,_|_|_.__/|_|_| |_|\__,_|\___|_|   



neuralbinder is a package to analyze the RNAbinding protein (RBP) specificity using datasets generated from high-throughput experiments.


* Documentation: <readthedocs link goes here>.


What is it?
-----------

**neuralbinder** consists of a bunch of routines to build and train neural network models, 
primarily using deepomics, which is a high-level tensorflow API for biological applications. 
NeuralBinder processes data using wrangler to process omics datasets into a format that is 
acceptable for inputs to a neural network. Different models can be found in the model zoo.

The package is flexible and can handle data from different kinds of high-throughput experiments
such as RNACompete, CLIP-Seq, e-CLIP etc. 

What can I use it for?
----------------------

Given a set of sequences that are believed to interact with your RBP of interest along with
other negative dataset consisting of sequences that do not interact with RBP, **neuralbinder**
employs a deep learning approach to identify potential RNA binding motif.

**neuralbinder** consists of following set of programs that can be readily used to train and 
test the performance of different neural network models. Currently the model zoo consists of 
convnet, all_convnet and residualbind models. 

============================      ==================
Script                            Description
============================      ==================
affinitybinder_train               Train on all the RNAcompete datasets
                                  
affinitybinder_test		  Evaluate the performance of pre-trained model 
                                  on test dataset derived from RNACompete experiment

nb_saliency_plot                  Given a model, saliency plot is generated for a 
                                  requested sequence from test dataset

clipbinder_train                     Train on a given clip dataset

clipbinder_test			  Evaluate the performance of pre-trained model on 
                                  a given test dataset derived from CLIP experiment
============================      ==================

Once all regions have been 'censused', the results may be sorted by one of four
mathematical operations: `max`, `min`, `median` and `mean`. So you may be interested
in subregions of your sequence(s) that feature the most missing nucleotides, or
subregions that contain the mean or median number of SNPs or the lowest GC-ratio.


Why should I use it?
--------------------

**neuralbinder** is hardly the first deep learning tool buit for analysis of biological
data, so why bother using it in your computational pipeline?

This tool does make building deep learning models with different architectures easy. The
framework could be readily used to test and deploy these algorithms using powerful
tensorflow platform on high-throughput datasets. It makes it easier to build an architecture,
train it on different types of high-throughput RNA binding protein datasets and infer the 
biologically relevant motifs associate with it.

The package is customisable and extendable, those even vaguely familiar with
Python should be able to buid a deep learning model to meet their requirements.

**neuralbinder** can be locally installed and can easily be trained and deployed 
specifically on a given RNA binding protein experiment.

Requirements
------------
To use;

* numpy
* scipy
* sklearn
* tensorflow
* matplotlib (for plotting)

To test;

* tox
* pytest

For coverage;

* nose
* python-coveralls

Installation
------------
From source distribution

::

    $ python setup.py install


Citation
--------

Please cite us so we can continue to make useful software! ::

    Koo., P., Anand. P., Eddy., S., (2017/2018). ResidualBind: Improved predictions of RNA binding protein specificities using deep learning. (Manuscript under preparation)
    
::

	@article{
	bibtex file
	}

License
-------
**neuralbinder** is distributed under the MIT license, see `LICENSE`_.

.. _LICENSE: https://raw.githubusercontent.com/p-anand/neuralbinder/master/LICENSE#hyperlink-targets
