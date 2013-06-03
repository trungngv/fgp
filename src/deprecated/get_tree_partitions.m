function partitions = get_tree_partitions(tree, X, Y)
%GET_TREE_PARTITIONS partitions = get_tree_partitions(tree, X, Y)
%
% Returns the partitions of the dataset by a regression tree trained with
% RegressionTree.fit. The output variables are cells where each element of
% the cell corresponds to a partition.
% 
% Usage:
%    partitions = GET_TREE_PARTITIONS(tree, X, Y)
%    partitions = GET_TREE_PARTITIONS(tree, X, [])
% or 
%    partitions = GET_TREE_PARTITIONS(tree)
%
% INPUT
%   - tree : a tree trained by RegressionTree.fit
%   - X, Y : input and output points to be partitioned (if either of X or Y
%   is missing, tree.X and tree.Y will be used instead)
%
% OUTPUT
%   - partitions : a cell of partitions
%
% Trung V. Nguyen
% 04/02/13
if nargin == 1
  X = tree.X;
  Y = tree.Y;
end

yhat = predict(tree, tree.X);
partitionMean = unique(yhat); % means of all leaf nodes
nParts = numel(partitionMean);
partitions = cell(nParts, 1);
yhat = predict(tree, X);
for i=1:nParts
  partitions{i}.x = X(yhat == partitionMean(i),:);
  if ~isempty(Y)
    partitions{i}.y = Y(yhat == partitionMean(i),:);
  end  
end

