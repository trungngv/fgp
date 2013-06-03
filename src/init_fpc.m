function inducing_ind = init_fpc(x,k,m)
%INIT_FPC clusters = init_fpc(x,k,m)
%   
% Inducing points init by farthest-point clustering. This uses the
% farthest-point clustering method, which clusters dataset x into k partitions.
% m inducing points are randomly selected from the resulting partitions.
% Note that this method is very similar to k-means.
%
% INPUT:
%   x : n x dim
%   k : num of partitions
%   m : num of inducing points per partitions
%
% OUTPUT:
%   clusters : m x k where column k contains indice of inducing points in k
[~,clusterInd,~,~] = KCenterClustering(size(x,2),size(x,1),x',k);
inducing_ind = zeros(m,k);
for i=0:k-1
  ind_i = find(clusterInd == i);
  inducing_ind(:,i+1) = ind_i(randperm(numel(ind_i),m));
end


