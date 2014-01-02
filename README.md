Fast Allocation of Gaussian Process Experts

Author: Trung V. Nguyen (trung.ngvan@gmail.com) and Edwin V. Bonilla

This is the package MSGP that implements the mixture of sparse Gaussian Process experts model in the paper 'Fast Allocation of Gaussian Process Experts'.

1. Datasets
-----------------------------
The 4 datasets (kin40k, pol, pumadyn32nm, and motorcycle) are provided in the 'data' directory.

For kin40k, pol, and pumadyn32nm, call load_data() to get the training and testing data used in the paper.

Example: [x,y,xtest,ytest] = load_data('data/kin40k','kin40k');

- The 100k songs dataset is not provided due to its large size but can be obtained at
 http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

To get the same training and testing data as in the paper, simply run:

src/scripts/song100kscript.m 

2. How to run MSGP
----------------------------
The two main functions for using MSGP are src/msgp_train.m and src/msgp_predict.m
See src/scripts/demo.m to see the example of using msgp on the motorcycle dataset.

3. Other baselines
----------------------------
The localFITC methods can be run with src/gps_fitc_train.m and src/gps_fitc_predict.m.

See src/scripts/randpar_batch.m for an example.

The SoD baseline is in src/scripts/song100k_sod.m

The GPSVI baseline is in src/gpsvi/batch_gpsvi_big.m (note that it requires the learned hyperparameters of SoD).

Other baselines for the 100k dataset can be found in src/scripts/song100k_baselines.m.

4. Multicore implementation
----------------------------
The main code for multicore is in libs/multicore/msgp. Email me if you have problems using multicore with MSGP.

5. Dependencies
----------------------------
- MSGP requires 3 external libraries: GPML [1], GPRApprox_Comparison [2], and SPGP_dist [3]
in addition to my own library of utility function (myutils)
- If multicore library is used, it also requires the multicore package
- If GPSVI is used, it requires the gpsvi implementation by Edwin Bonilla which depends on the 
netlab and myutil package
All libraries are included for your convenience.

References:
----------------------------
[1] Rasmussen, Carl Edward and Nickisch, Hannes. Gaussian processes for machine learning (gpml) toolbox. The
Journal of Machine Learning Research, 11:3011–3015, 2010

[2] Chalupka, Krzysztof, Williams, Christopher KI, and Murray, Iain. A framework for evaluating approximation
methods for gaussian process regression. arXiv preprint arXiv:1205.6326, 2012.

[3] Snelson, Ed and Ghahramani, Zoubin. Sparse gaussian processes using pseudo-inputs, NIPS 2006.

