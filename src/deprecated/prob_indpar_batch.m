% demo of inducing (basis) based partition
clear; clc; close all;
theseed = 1110;

datasetsDir = '/home/trung/projects/datasets/';
datasetNames = {'kin40k','pumadyn32nm','myelevators','pol'};
output_file = '/home/trung/projects/ensemblegp/output/probindpark3.mat';
try
  load(output_file, 'logger');
catch eee
  logger = [];
  save(output_file, 'logger');
end

max_fevals = 1;
numz = 2;
numzstar = 2;
for sss=1:5
  seed = theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  timestamp = [strdate 't' num2str(tic)];

  for K=3:3
    num_inducing = 1500/K;
    for idata = 1:length(datasetNames)
      name = [datasetNames{idata} 'Nu' num2str(num_inducing) 'k' num2str(K)];
      logger.(timestamp).rng = seed;
      logger.(timestamp).(name).version = 1;
      logger.(timestamp).(name).K = K;
      logger.(timestamp).(name).num_basis = num_inducing;
      logger.(timestamp).(name).num_samples = numz;
      logger.(timestamp).(name).num_test_samples = numzstar;
      logger.(timestamp).(name).max_fevals = max_fevals;

      [x,y,xt,yt] = load_data([datasetsDir datasetNames{idata}], datasetNames{idata});
      dim = size(x,2);

      % init hyper-parameters
      %TODO: use some partitioning to init xu
      [~,randind] = sort(rand(size(x,1),1));
      randind = randind(1:(num_inducing*K));
      randind = reshape(randind,num_inducing,K);
      hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y,false));
      w0 = [];
      for k=1:K
        xu_k = x(randind(:,k),:);
        w0 = [w0; xu_k(:)];
        % TODO: different scale for each partition?
        w0 = [w0; hyp];
      end
      % learn basis points and hyp
      tstart = tic;
      [w,fval] = minimize(w0,'prob_indpar_marginal',-max_fevals,x,y,K,num_inducing,numz);
      logger.(timestamp).(name).training_time = toc(tstart);
      logger.(timestamp).(name).obj = fval(end);
      logger.(timestamp).(name).optim_w = w;
      
      % assign points to partition based on the basis points
      W = reshape(w,numel(w)/K,K);
      [fmean,logpred,valid] = prob_indpar_predict(x,y,xt,yt,W,num_inducing,numz,numzstar);
      logger.(timestamp).(name).valid = valid;
      logger.(timestamp).(name).prediction_time = toc(tstart);
      fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
      fprintf('nlpd = %.4f\n', -mean(logpred));
      logger.(timestamp).(name).smse = mysmse(yt,fmean,mean(y));
      logger.(timestamp).(name).nlpd = -mean(logpred);
      save(output_file, 'logger');
    end
  end
end

