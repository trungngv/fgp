function [ymu, yvar] = gp_predict_weighted(models, partitions, Xtest)
%PREDICTMULTIPLEGPS [ymu, yvar] = [ymu, yvar] = gp_predict_weighted(models, partitions, Xtest)
%   Weighted prediction using multiple GPs.
%
% INPUT
%   - models : trained GP models for all partitions
%   - partitions : normalised input and outputs of all partitions
%   - Xtest : test input points
%
% OUTPUT
%   - Ymean, Yvar
%
% Trung V. Nguyen
% 14/01/13
covfunc = {@covSEard}; likfunc = @likGauss; infFunc = @infExact;

% Must normalize data to be consistent with training procedure
nPartitions = size(partitions,1);
yMeans = zeros(size(Xtest,1),size(models,1)); % Ntest x M
yVars = yMeans;
for i=1:nPartitions
  [~, yVars(:,i), yMeans(:,i)] = gp(models{i}.hyp, infFunc, [], covfunc, likfunc, ...,
    partitions{i}.X, partitions{i}.Y, standardize(Xtest,1,partitions{i}.xmean, partitions{i}.xstd));
  yMeans(:,i) = yMeans(:,i) .* partitions{i}.ystd + partitions{i}.ymean;
end

% maxVariance = max(yVars, [], 2);
% maxVariance = repmat(maxVariance,1,size(yVars,2));
% yVars(yVars == maxVariance) = Inf;
% sumInverseVariance = sum(1./yVars, 2); % norm constant of weights
% ymu = sum(yMeans./yVars, 2) ./ sumInverseVariance; % normalized iinverse variance

sumInverseVariance = sum(1./yVars, 2); % norm constant of weights
ymu = sum(yMeans./yVars, 2) ./ sumInverseVariance; % normalized iinverse variance

%ymu = sum(yMeans./yVars, 2); % inverse variance
yvar = zeros(size(ymu)); % TODO
end

