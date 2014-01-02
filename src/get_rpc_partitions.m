function [centroids,label] = get_rpc_partitions(x,k)
%GET_RPC_PARTITIONS [centroids,label] = get_rpc_partitions(x,k)
%   Partition the dataset using rpc.
%
label = rpClust(x,floor(size(x,1)/k));
centroids = zeros(k,size(x,2));
for i=1:k
  centroids(i,:) = mean(x(label == k,:));
end

