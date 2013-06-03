function [ulabel,xlabel,xt_label] = zmap_mahala_em(U,K,x,xt)
%ZMAP_MAHALA_EM [ulabel,gmfit] = zmap_mahala_em(U,K,x,xt)
%   
% Compute maximum a posteriori (MAP) of label assignment z_{MAP} using
% shared diagonal covariance matrix. This is equivalent to assigning x
% to the nearest center measured by mahalanobis distance. Unlike
% ZMAP_MAHALA, the labels of partitions are unknown so must be learned
% using Expectation Maximization algorithm.
%
% label(x_i) = argmin_k mahalanobis_dist(x,center_k) 
% where center_k = the center (mean) of all basis points in partition k.
%
% INPUT
%   - U : M*D
%   - K : number of partitions
%   - x : training inputs (or [])
%   - xt : test inputs (or [])
%
% OUTPUT
%   - ulabel : labels of inducing inputs
%   
%
% Trung Nguyen
% 05/2013
% make sure em converges to same optima given same inputs
rng(0212,'twister');
options.maxiter = 100;
gmfit = gmdistribution.fit(U,K,'CovType','diagonal','SharedCov',true,...
  'Regularize',1e-6,'options',options);
ulabel = gmfit.cluster(U);
% monitor
disp(histc(ulabel,1:K)');
xlabel = []; xt_label = [];
if ~isempty(x)
  xlabel = gmfit.cluster(x);
end
if ~isempty(xt)
  xt_label = gmfit.cluster(xt);
end

