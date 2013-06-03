function [centroids,label] = get_random_partitions(x,k)
  n = size(x,1);
  label = randsample(1:k,n,true);
  centroids = zeros(k,size(x,2));
  for i=1:k
    centroids(i,:) = mean(x(label == k,:));
  end
