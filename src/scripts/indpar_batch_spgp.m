% demo of inducing (basis) based partition
clear; clc; close all;
theseed = 1111;

func_assign = 'assign_nearest_center';
%func_assign = @assign_test;
datasets_dir = 'fagpe/data/';
output_file = 'fagpe/output/fitc_spgp.mat';
% datasetName = 'mysynth';
% fileName = 'synth1';
dataset_names = {'kin40k','pumadyn32nm','myelevators','pol'};
try
  load(output_file, 'logger');
catch eee
  logger = [];
  save(output_file, 'logger');
end

nu = 1500;
K = 1;
max_fevals = 1000;
for sss=0:4
  seed=theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  tstamp = [strdate 't' num2str(tic)];
  for idata = 1:length(dataset_names)
    name = dataset_names{idata};
    logger.(tstamp).rng = seed;
    logger.(tstamp).(name).version = 1;
    logger.(tstamp).(name).K = K;
    logger.(tstamp).(name).num_basis = nu;
    logger.(tstamp).(name).func_assign = func_assign;
    logger.(tstamp).(name).max_fevals = max_fevals;

    [x,y,xt,yt] = load_data([datasets_dir dataset_names{idata}], dataset_names{idata});
    dim = size(x,2);

    % init hyper-parameters
    %TODO: use some partitioning to init xu
    [~,randind] = sort(rand(size(x,1),1));
    randind = randind(1:(nu*K));
    randind = reshape(randind,nu,K);
    hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y,false));
    w0 = [];
    for k=1:K
      xu_k = x(randind(:,k),:);
      w0 = [w0; xu_k(:); hyp];
    end
    % learn basis points and hyp
    tstart = tic;
    [w,fval] = minimize(w0,'indpar_marginal',-max_fevals,x,y,K,nu,func_assign);
    logger.(tstamp).(name).training_time = toc(tstart);
    logger.(tstamp).(name).obj = fval(end);
    logger.(tstamp).(name).optim_w = w;

    % assign points to partition based on the basis points
    tstart = tic;
    W = reshape(w,numel(w)/K,K);
    label = feval(func_assign,W(1:end-dim-2,:),x,nu);
    test_label = feval(func_assign,W(1:end-dim-2,:),xt,nu);

    % predict
    Nt = size(xt,1);
    valid = true;
    fmean = zeros(Nt,1); logpred = zeros(Nt,1);
    for k=1:K
      indk = label == k;
      test_indk = test_label == k;
      wk = W(:,k);
      %TODO: use a unboxing function
      xu = reshape(wk(1:end-dim-2),nu,dim);
      hypk = fitc_hyp_to_gpml(wk,dim);
      try
        [fmean(test_indk),~,logpred(test_indk)] = gpmlFITC(x(indk,:),...
          y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,false);
      catch eee
        valid = false;
      end
    end
    logger.(tstamp).(name).prediction_time = toc(tstart);
    logger.(tstamp).(name).valid = valid;
    logger.(tstamp).(name).smse = mysmse(yt,fmean,mean(y));
    logger.(tstamp).(name).nlpd = -mean(logpred);
    logger.(tstamp).(name).mae = mean(abs(fmean-yt));
    logger.(tstamp).(name).sqdiff = sqrt(mean((fmean-yt).^2));
    fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
    fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
    fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2))); 
    fprintf('nlpd = %.4f\n', -mean(logpred));
    save(output_file, 'logger');
  end
end
