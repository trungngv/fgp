% for 100k dataset
clear; close all; clc;
theseed = 20;

datasets_dir = 'data/';
dataset_names = {'song100k'};
output_file = 'output/song100k_rpc.mat';
logger = [];

max_fevals = 1000;
zeromean = true;
for sss=1:5
  seed=theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  tstamp = [strdate 't' num2str(tic)];
  for K=[20]
    %num_inducing = 1500/K;
    num_inducing = 300;
    for idata = 1:length(dataset_names)
      try    load(output_file, 'logger'); 
      catch eee,    save(output_file, 'logger');
      end
      logger.(tstamp).rng = seed;
      logger.(tstamp).version = 1;
      logger.(tstamp).K = K;
      logger.(tstamp).zeromean = zeromean;
      logger.(tstamp).num_inducing = num_inducing;
      logger.(tstamp).max_fevals = max_fevals;

      [x,y,xt,yt] = load_data([datasets_dir dataset_names{idata}], dataset_names{idata});
      y0 = y - mean(y);
      dim = size(x,2);

      tstart = tic;
      [centroids,label] = get_rpc_partitions(x,K);
      disp('training fitc models')
      if zeromean
        models = gps_fitc_train(x,y0,K,label,num_inducing,max_fevals);
      else  
        models = gps_fitc_train(x,y,K,label,num_inducing,max_fevals);
      end  
      logger.(tstamp).training_time = toc(tstart);
      logger.(tstamp).optim_models = models;
      
      tstart = tic;
      [~,test_label] = get_kmeans_partitions(xt,K,centroids);
      [fmean,~,logpred,valid] = gps_fitc_predict(models,x,y,xt,yt,label,...
        test_label,zeromean);
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
      save(output_file, 'logger');
    end
  end
end

