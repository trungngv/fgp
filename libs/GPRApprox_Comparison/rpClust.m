function [cIds splits splitAxes] = rpClust(data, cSize, varargin)
% Recursive Projection Clustering
%
% For datapoints in rows of 'data', split
% recursively into clusters of equal size by 
% choosing two points at random, and splitting
% all the points into two equal-size clusters by
% their projection on the line connecting the two
% points. Continue recursively until all the clusters
% are of size at most cSize.
%
% varargins are only used internally to store data indices.
%
% Conceived by Iain Murray, implemented by Krzysztof Chalupka,
% University of Edinburgh, 2011
global splits
global splitAxes
firstIter = 0;
[N, D] = size(data);
if size(varargin, 2) == 0
  allIds = 1:N;
  firstIter = 1;
  treeNodeId = 1;
  splits = zeros(1, ceil(size(data,1)/cSize));
  splitAxes = zeros(size(data,2), ceil(size(data,1)/cSize));
else
  allIds = varargin{1};
  assert(N == length(allIds));
  treeNodeId = varargin{2};
end
if N  <= cSize
  if ~firstIter
    cIds = {allIds};
  else
    cIds = zeros(1,N);
  end
  return
end

% Choose a random axis along two datapoints.
r1 = randi(N);
r2 = r1;
while r1==r2; r2 = randi(N); end;
axisVec = (data(r1,:) - data(r2,:))';

% Split data in two by projection coefficients.
projCoeffs= dot(repmat(axisVec, 1, N), data');
mdn = median(projCoeffs);
splits(treeNodeId) = mdn;
splitAxes(:,treeNodeId) = axisVec;
left = find(projCoeffs >= mdn);
right = find(projCoeffs < mdn);

treeLvl = floor(log2(treeNodeId));
treeIncrmt = treeNodeId - 2^treeLvl;
child1 = 2^(treeLvl+1)+2*treeIncrmt;
child2 = child1+1;
% Repeat recursively.
clustered = [rpClust(data(left,:), cSize, allIds(left), child1), ...
	     rpClust(data(right,:), cSize, allIds(right), child2)];
if ~firstIter
  cIds = clustered;
else
  % convert the cell array with grouped indices into a list of cluster indices.
  cIds = zeros(1,N);
  ccs  = []; 
  for i=1:length(clustered)
    for j=1:length(clustered{i})
      cIds(clustered{i}(j)) = i-1;
    end
    ccs = [ccs mean(data(clustered{i},:),1)'];
  end
end
