function models = gps_fitc_train(x,y,K,label,num_inducing,max_fevals)
%GPS_FITC_TRAIN models = gps_fitc_train(x,y,K,label,num_inducing,max_fevals)
%
%   Trains multiple Gaussian process regression models corresponding to the
%   partitions given in PARTITIONS. Each partition is fitted using the FITC
%   model. If there are M partitions, M models will be returned.
%
%   NOTE: GPS_FITC_TRAIN does not perform any pre-processing of inputs or
%   outputs.
%
%   Structure of a trained GP model:
%   - model.hyp     : the learned hyp structure
%   - model.nlm     : the negative log marginal likelihood of the model
% 
% INPUT
%   - label : partition/cluster label of the training data
%   - num_inducing : number of inducing points to use for each partition
%
% OUTPUT
%   - models: a cell where each element is a trained fitc model. 
%
% Trung V. Nguyen
% 26/03/13

%matlabpool(4);
models = cell(K,1);
D = size(x,2);
for k=1:K
  fprintf('training partition %d\n', k);
  xk = x(label == k,:);
  yk = y(label == k);
  if (size(xk,1) > num_inducing)
    gpml_hyp = gpml_init_hyp(xk,yk,false);
    [dum,randind] = sort(rand(size(xk,1),1)); clear dum;
    randind = randind(1:num_inducing);
    xu_init = xk(randind,:);
    w_init = [reshape(xu_init,num_inducing*D,1);gpml_hyp_to_fitc(gpml_hyp)];
    [w, fw] = minimize(w_init,'spgp_lik',-max_fevals,yk,xk,num_inducing);
    models{k}.obj = fw(end);
    models{k}.xu = reshape(w(1:num_inducing*D,1),num_inducing,D);
    models{k}.hyp = fitc_hyp_to_gpml(w,D);
    models{k}.nlm = fw(end);
  end  
end

