function fitc_hyp = gpml_hyp_to_fitc(gpml_hyp)
%GPML_HYP_TO_FITC fitc_hyp = gpml_hyp_to_fitc(gpml_hyp)
%   Converts the gpml parametrisation of hyp into that of FITC (SPGP).
%
% Trung V. Nguyen
% 26/03/13
fitc_hyp = [-2*gpml_hyp.cov(1:end-1); 2*gpml_hyp.cov(end); 2*gpml_hyp.lik];


