function [label,centers,dist,detD] = zhat_mahala(W,x,num_inducing,centers)
%ZMAP_MAHALANOBIS [label,centers,dist,detD] = zhat_mahala(W,x,num_inducing,centers)
%   
% Compute the fast allocation zhat using shared diagonal covariance matrix.
% This is equivalent to assigning x to the nearest center measured by mahalanobis distance.
%
% label(x_i) = argmin_k mahalanobis_dist(x,center_k) 
% where center_k = the center (mean) of all basis points in partition k.
%
% INPUT
%   - W : (nu*D) x K 
%   - x : N x D
%   - num_inducing : 
%   - centres : optional (K x D)
% OUTPUT
%   - label : assignment of point to partitions
%   - centers
%
% Trung Nguyen
% 04/2013
K = size(W,2);
dim = size(x,2);
if nargin == 3   % centres not given
  centers = partition_centers(W,num_inducing,K,dim);
end  

Mu = zeros(size(W));
for k=1:K
  matMu = repmat(centers(k,:),num_inducing,1); % nu x dim
  Mu(:,k) = matMu(:); % (nu*dim)x1 as W(:,k)
end
% sum over partitions and then reshape
diagCov = reshape(sum((W-Mu).^2,2),num_inducing,dim);
% sum over xu and normalize (1xdim)
diagCov = sum(diagCov)/(num_inducing*K-K);
diagCov = diagCov.^ 0.5;
detD = prod(diagCov).^2;

nx = (x ./ repmat(diagCov,size(x,1),1))';
nmu = (centers ./ repmat(diagCov,K,1))';
% (x-mu)^T V^{-1} (x-mu) = (x-mu)^T Lambda (x-mu)
%                            = (Lamba^(1/2)(x-mu))^T (Lamba^(1/2)(x-mu)) 
% N x K distance from each point to partition centers
dist = sq_dist(nx,nmu);
[~,label] = min(dist,[],2);

