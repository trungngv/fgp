function [ci cc] = rrClust(X, cSize, varargin)

% RRCLUST - random (recursive) clustering. 
% Cluster the data by choosing the cluster centers
% at random and assigning the closest points to each
% cluster. Repeat recursively until all the clusters
% are of size at most cSize.
%
% [ci cc] = rrClust(X, cSize)
%
% X  -  an N x D matrix of D dimensional datapoints.
% cSize   -  upper bound on the cluster size. The algorithm
%            will start by choosing floor(N/cSize) cluster 
%            centers.
% Author - Krzysztof Chalupka, University of Edinburgh 2010.

[N, D] = size(X);
cc = [];
% If recursive step, split in two.
if size(varargin,1)==1
  firstLevel=0;
  m = 2;
  clustCount = max(varargin{1})+1;
% If first iteration, split into clusters of size ~cSize.
else
  firstLevel=1;
  m = floor(N/cSize);
  clustCount=0;
end

cIds = randperm(N);
cc = X(cIds(1:m),:);
ci = clustCount+allDists(cc, X, 100)-1;

for i=(clustCount+[0:m-1])
  currentIds = find(ci==i);
  if length(currentIds) > cSize
    [recCi recCC] = rrClust(X(currentIds,:), cSize, unique(ci));
    for i=unique(recCi)
      assert(length(find(recCi==i)) <= cSize, 'assertion broken in rrClust.m: cluster too large!');
    end
    ci(currentIds) = recCi;
  end
end

if size(varargin,1)==0
  % Recursion done. Ensure the cluster ids
  % range from 0 to the number of clusters.
  clustIds = unique(ci);
  cc=[];
  for i=1:length(clustIds)
    ci(find(ci==clustIds(i))) = i-1;
    cc = [cc mean(X(find(ci==i-1),:),1)'];
  end
end
