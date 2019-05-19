# ==Convolutional Poisson Gamma Belief Network==
This is code for the paper "Convolutional Poisson Gamma Belief Network" published in ICML2019 

Created by Chaojie Wang , Bo Chen , Sucheng Xiao at Xidian University and Mingyuan Zhou at University of Texas at Austin
, https://arxiv.org/abs/1905.05394

# ==Requirement==
Tensorflow >= 1.0

PyCUDA >= 0.8 

PyCUDA can be download from following address https://mathema.tician.de/software/pycuda/

# ==Data Source==
All data source files can be found in following addresses and have been included in our repository

- MR: https://www.cs.cornell.edu/people/pabo/movie-review-data/

- TREC: http://cogcomp.cs.illinois.edu/Data/QA/QC/

- SUBJ: http://www.cs.cornell.edu/people/pabo/movie-review-data/

- ELEC: https://github.com/riejohnson/ConText

- IMDB: http://ai.stanford.edu/~amaas/data/sentiment/

# ==Overview==
- CPFA_Mnist_Demo folder contains 4 different training algorithms for CPFA, including Toeplitz, Element, Element-Parallel and SGMCMC mehthods

- CPGBN_Text_Demo folder contains Datasets and experiment code to reproduce the results in our paper

- CPGBN_Derivation_Draft file provides a detailed derivation for the CPGBN

# ==Citations==
If you find that the algorithms in this repository are useful for your research, please refer to the following article:

@inproceedings{CPGBN_ICML2019,<br>
title={{C}onvolutional {P}oisson {G}amma {B}elief {N}etwork},<br>
author={Chaojie Wang and Bo Chen and Sucheng Xiao and Mingyuan Zhou}, booktitle={ICML}, year={2019}}

# ==Contact== 
Contact Bo Chen bchen@mail.xidian.edu.cn or Chaojie Wang xd_silly@163.com

Copyright (c), 2018, Chaojie Wang xd_silly@163.com
