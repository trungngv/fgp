function gpml_hyp = fitc_hyp_to_gpml(fitc_hyp,dim)
%FITC_HYP_TO_GPML gpml_hyp = fitc_hyp_to_gpml(fitc_hyp,dim)
%   Convert the FITC (SPGP) parametrisation of hyperparameters to that of GPML.
%
% Trung V. Nguyen
% 26/03/13
gpml_hyp.cov = [-0.5*fitc_hyp(end-dim-1:end-2); 0.5*fitc_hyp(end-1)];
gpml_hyp.lik = 0.5*fitc_hyp(end);

