function inducing_ind = init_kmeans(x,k,m)
%INIT_KMEANS clusters = init_kmeans(x,k,m)
%   
% Inducing points init by kmeans clustering. This uses the
% kmeans clustering method, which clusters dataset x into k partitions.
% m inducing points are randomly selected from the resulting partitions.
%
% INPUT:
%   x : n x dim
%   k : num of partitions
%   m : num of inducing points per partitions
%
% OUTPUT:
%   inducing_ind : m x k where column k contains indice of inducing points in k
[clusterInd,~] = kmeans(x,k);
inducing_ind = zeros(m,k);
for i=1:k
  ind_i = find(clusterInd == i);
  inducing_ind(:,i) = ind_i(randperm(numel(ind_i),m));
end


