% demo of inducing (basis) based partition
clear; clc; close all;
theseed = 20;

func_assign = 'zmap_mahala';
%func_assign = @assign_test;
datasets_dir = 'fagpe/data/';
output_file = 'fagpe/output/100k_msgpv2_zhat.mat';
dataset = 'song100k';
zeromean = true;
try
  load(output_file, 'logger');
catch eee
  logger = [];
  save(output_file, 'logger');
end

[x,y,xt,yt] = load_data([datasets_dir dataset], dataset);
if zeromean,
  y0 = y-mean(y);
else
  y0 = y;
end
disp('finished reading data')

dim = size(x,2);
K = 20;
nu = 300;
max_iters = 20;
hyp_iters = 10;
for sss=1:5
  % should load logger here!
  seed=theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  tstamp = [strdate 't' num2str(tic)];
  logger.(tstamp).rng = seed;
  logger.(tstamp).version = 2;
  logger.(tstamp).K = K;
  logger.(tstamp).num_basis = nu;
  logger.(tstamp).func_assign = func_assign;
  logger.(tstamp).max_iters = max_iters;
  logger.(tstamp).hyp_iters = hyp_iters;
  logger.(tstamp).zeromean = zeromean;
  
  % init hyper-parameters
  randind = init_rpc(x,K,nu);
  hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y0,false));
  w0 = [];
  for k=1:K
    xu_k = x(randind(:,k),:);
    %w0 = [w0; xu_k(:); hyp];
    w0 = [w0; xu_k(:); gpml_hyp_to_fitc(gpml_init_hyp(xu_k,y0(randind(:,k)),false))];
  end
  
  % learn basis points and hyp
  tstart = tic;
  [w,fval,label] = msgp_train(w0,x,y0,K,nu,max_iters,hyp_iters);
  disp('finished learning hyperparameters');
  logger.(tstamp).training_time = toc(tstart);
  logger.(tstamp).obj = fval;
  logger.(tstamp).optim_w = w;
  

  % assign points to partition based on the basis points
  W = reshape(w,numel(w)/K,K);
  test_label = feval(func_assign,W(1:end-dim-2,:),xt,nu);

  % predict
  Nt = size(xt,1);
  valid = true;
  fmean = zeros(Nt,1); logpred = zeros(Nt,1);
  tstart = tic;
  for k=1:K
    indk = label == k;
    test_indk = test_label == k;
    wk = W(:,k);
    %TODO: use a unboxing function
    xu = reshape(wk(1:end-dim-2),nu,dim);
    hypk = fitc_hyp_to_gpml(wk,dim);
    try
      [fmean(test_indk),~,logpred(test_indk)] = gpmlFITC(x(indk,:),...
        y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,zeromean);
    catch eee
      valid = false;
      fmean(test_indk) = repmat(mean(y(indk)),sum(test_indk),1);
      fprintf('mean: %.1f\t num = %d\n', mean(y(indk)), sum(test_indk));
    end
  end
  logger.(tstamp).prediction_time = toc(tstart);
  logger.(tstamp).valid = valid;
  logger.(tstamp).smse = mysmse(yt,fmean,mean(y));
  logger.(tstamp).nlpd = -mean(logpred);
  logger.(tstamp).mae = mean(abs(fmean-yt));
  logger.(tstamp).sqdiff = sqrt(mean((fmean-yt).^2));
  fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
  fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
  fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2))); 
  fprintf('nlpd = %.4f\n', -mean(logpred));
  %figure; hist(abs(fmean-yt),200);
  save(output_file, 'logger');
end

