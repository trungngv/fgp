function [yMeans, yVars] = gp_predict_hard_assignment(models, partitions)
%PREDICTHARDASSIGNMENT [yMeans, yVars] = gp_predict_hard_assignment(models, partitions)
%   Prediction made by hard assignment of points to clusters.
%
% Trung V. Nguyen
% 04/02/13
covfunc = {@covSEard}; likfunc = @likGauss; infFunc = @infExact;
nParts = size(partitions,1);
yMeans = cell(size(partitions)); % Ntest x M
yVars = yMeans;
%TODO if XtestParts{i} has elements
for i=1:nParts
  [~, yVars{i}, yMeans{i}] = gp(models{i}.hyp, infFunc, [], covfunc, likfunc, ...,
    partitions{i}.X, partitions{i}.Y, standardize(partitions{i}.Xtest,1,partitions{i}.xmean, partitions{i}.xstd));
  yMeans{i} = yMeans{i} .* partitions{i}.ystd + partitions{i}.ymean;
end
end

