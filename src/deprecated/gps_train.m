function models = gps_train(initModel, partitions)
%GPS_TRAIN gps_train(initModel, partitions)
%
%   Trains multiple Gaussian process regression models corresponding to the
%   partitions given in partitions. If there are M partitions, M models
%   will be returned.
%
%   NOTE: TRAIN_GPS does not perform automatic normalisation of
%   inputs and outputs.
%
%   Structure of a trained GP model:
%   - model.inithyp : the initialized hyp structure (same as in gpml by Carl Rasmussen)
%   - model.hyp     : the learned hyp structure
%   - model.nlm     : the negative log marginal likelihood of the model
% 
% INPUT
%   - initModel : an init model as the initialisation for this model (empty
%   if not available)
%   - partitions (cell) : a cell where each element contains a structure
%   for a partition
%       X : input points
%       Y : output points
%       svX : input support vectors (if using a trained svc model)
%       svY : output support vectors (if using a trained svc model)
%
% OUTPUT
%   - models: a cell where each element is a trained GP model. 
%
% Trung V. Nguyen
% 14/01/13
%
covfunc = {@covSEard}; likfunc = @likGauss; infFunc = @infExact;

%matlabpool(4);
numPartitions = numel(partitions);
models = cell(size(partitions));
D = size(partitions{1}.X,2);
for i=1:numPartitions
  if ~isempty(initModel)
    models{i}.inithyp = initModel.hyp;
  else
    models{i}.inithyp.cov = [rand(D,1); 0];
    models{i}.inithyp.lik = rand;
  end
  if isfield(partitions{i}, 'svX')
    [models{i}.hyp, models{i}.nlm] = minimize(models{i}.inithyp, @gp, 1000, ...
      infFunc, [], covfunc, likfunc, partitions{i}.svX, partitions{i}.svY);
  else
    [models{i}.hyp, models{i}.nlm] = minimize(models{i}.inithyp, @gp, 1000, ...
      infFunc, [], covfunc, likfunc, partitions{i}.X, partitions{i}.Y);
  end  
  models{i}.nlm = models{i}.nlm(end);
end

%predictive distribution
%[ystaaar s2 ystar] = gp(hyp, infFunc, [], covfunc, likfunc, X, Y, xtest);

end

