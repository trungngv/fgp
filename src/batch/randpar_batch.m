clear; close all; clc;
theseed = 1111;

datasets_dir = 'fagpe/data/';
dataset_names = {'kin40k','pumadyn32nm','myelevators','pol'};
%dataset_names = {'song100k'};
output_file = 'fagpe/output/randpar.mat';
logger = [];

max_fevals = 1000;
zeromean = false;
for sss=1:5
  seed=theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  tstamp = [strdate 't' num2str(tic)];
  for K=[3,2]
    num_inducing = 1500/K;
    %num_inducing = 300;
    for idata = 1:length(dataset_names)
      try    load(output_file, 'logger'); 
      catch eee,    save(output_file, 'logger'); end
      name = [dataset_names{idata} 'NU' num2str(num_inducing) 'k' num2str(K)];
      logger.(tstamp).rng = seed;
      logger.(tstamp).(name).version = 1;
      logger.(tstamp).(name).K = K;
      logger.(tstamp).(name).zeromean = zeromean;
      logger.(tstamp).(name).num_basis = num_inducing;
      logger.(tstamp).(name).max_fevals = max_fevals;

      [x,y,xt,yt] = load_data([datasets_dir dataset_names{idata}], dataset_names{idata});
      y0 = y - mean(y);
      dim = size(x,2);

      tstart = tic;
      [centroids,label] = get_random_partitions(x,K);
      disp('training fitc models')
      if zeromean
        models = gps_fitc_train(x,y0,K,label,num_inducing,max_fevals);
      else  
        models = gps_fitc_train(x,y,K,label,num_inducing,max_fevals);
      end  
      logger.(tstamp).(name).training_time = toc(tstart);
      logger.(tstamp).(name).optim_models = models;
      
      tstart = tic;
      [~,test_label] = get_kmeans_partitions(xt,K,centroids);
      [fmean,~,logpred,valid] = gps_fitc_predict(models,x,y,xt,yt,label,...
        test_label,zeromean);
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
end

