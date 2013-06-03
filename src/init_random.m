function inducing_ind = init_random(x,k,m)
%INIT_RANDOM inducing_ind = init_random(x,k,m)
%   
% Inducing points init by random. 
%
% INPUT:
%   x : n x dim
%   k : num of partitions
%   m : num of inducing points per partitions
%
% OUTPUT:
%   inducing_ind : m x k where column k contains indice of inducing points in k

[~,inducing_ind] = sort(rand(size(x,1),1));
inducing_ind = inducing_ind(1:(m*k));
inducing_ind = reshape(inducing_ind,m,k);

