# ResidualBind package overview

The ResidualBind is a python package that uses TensorFlow for DNN training and model interpretability with global importance analysis.


#### Dependencies
- Python 3.5 or greater
- Pandas, NumPy, SciPy, Matplotlib, H5py
- TensorFlow 1.15 or greater
- Logomaker (Tareen and Kinney)

#### Source files
- residualbind.py - class for ResidualBind and GlobalImportance 
- helper.py - functions to file handling
- explain.py - functions for in silico mutagenesis and k-mer alignments for motif visualization
- E_RNAplfold, H_RNAplfold, I_RNAplfold, M_RNAplfold - RNAplfold scripts to calculate probability of external loop, hairpin loop, internal loop, and multi-loop, respectively

#### Example files
- generate_rnacompete_2013_dataset.py - script to process the RNAcompete dataset
- train_rnacompete_2013.py - train a ResidualBind model on all RNAcompete experiments
- test_rnacompete_2013.py - test each ResidualBind model on all RNAcompete experiments
- global_importance_analysis.py - run GIA experiments systematically across all RNAcompete
- Figure1_performance_analysis.ipynb - jupyter notebook that generates Figure 1 in (Koo et al.)
- Figure2_RBFOX1_analysis.ipynb - jupyter notebook that generates Figure 2 in (Koo et al.)
- Figure3_VTS1_analysis.ipynb - jupyter notebook that generates Figure 3 in (Koo et al.)
- Figure4_GC-bias_analysis.ipynb - jupyter notebook that generates Figure 4 in (Koo et al.)

