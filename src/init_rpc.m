function inducing_ind = init_rpc(x,k,m)
%INIT_RPC inducing_ind = init_rpc(x,k,m)
%   
% Inducing points init by recursive projection clustering. This seems to
% generate more balanced size clusters.
%
% INPUT:
%   x : n x dim
%   k : num of partitions
%   m : num of inducing points per partitions
%
% OUTPUT:
%   inducing_ind : m x k where column k contains indice of inducing points in k
if size(x,2) == 1
  x = [x, x]; % trick for dim = 1 due to bug in the code
end
[clusterInd] = rpClust(x,floor(size(x,1)/k)); % equal cluster size
inducing_ind = zeros(m,k);
for i=0:k-1
  ind_i = find(clusterInd == i);
  inducing_ind(:,i+1) = ind_i(randperm(numel(ind_i),m));
end

