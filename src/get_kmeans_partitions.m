function [centroids,label] = get_kmeans_partitions(x,k,centroids)
%GET_KMEANS_PARTITIONS [centroids,label] = get_kmeans_partitions(x,k,centroids)
%   Partition the dataset using kmeans.
%
if ~isempty(centroids) % partition test set
  dist = sq_dist(x',centroids');
  [~,label] = min(dist,[],2);
else      % partition training set
  [label,centroids] = kmeans(x,k,'MaxIter',500);
end


