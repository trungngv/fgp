function [label,centers] = zhat_euclid(W,x,num_inducing,centers)
%ZMAP_EUCLID [label,centres] = zhat_euclid(W,x,num_inducing,centers)
%   
% Compute fast allocation zhat using shared isotropic covariance matrix.
% This is equivalent to assigning x to the nearest center measured by euclidean distance.
%
% label(x_i) = argmin_k euclidean_dist(x,center_k) 
% where center_k = the center (mean) of all basis points in partition k.
%
% INPUT
%   - W : (nu*D) x K 
%   - x 
%   - nu : 
%   - centres : optional
% OUTPUT
%   - label : assignment of point to partitions
%
K = size(W,2);
dim = size(x,2);
if nargin == 3   % centres not given
  centers = partition_centers(W,num_inducing,K,dim);
end  
 % N x K distance from each point to partition centers
dist = sq_dist(x',centers');
[~,label] = min(dist,[],2);

