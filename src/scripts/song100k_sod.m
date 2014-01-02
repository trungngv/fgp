% for 100k dataset
clear; close all; clc;
theseed = 20;
disp('start batches...')
datasets_dir = 'data/';
dataset_name = 'song100k';
output_file = 'output/song100k_sod.mat';
logger = [];

max_fevals = 1000;
zeromean = true;
subset_size = 2000;
for sss=1:5
  seed=theseed+sss;
  rng(seed,'twister');
  strdate = datestr(now, 'mmmdd');
  tstamp = [strdate 't' num2str(tic)];

  try    load(output_file, 'logger'); 
  catch eee,    save(output_file, 'logger');   end
  logger.(tstamp).rng = seed;
	logger.(tstamp).zeromean = zeromean;
	logger.(tstamp).subset_size = subset_size;
  logger.(tstamp).max_fevals = max_fevals;

	[x,y,xt,yt] = load_data([datasets_dir dataset_name], dataset_name);
	sub_ind = randperm(size(x,1),subset_size);
	x = x(sub_ind,:);
	y = y(sub_ind,:);
	dim = size(x,2);

	tstart = tic;
	disp('training standard gp model')
  if zeromean
    y0 = y-mean(y); % zero-mean the outputs
  else
    y0 = y;
  end

  lengthscales = log((max(x)-min(x))'/2);
  lengthscales(lengthscales<-1e2)=-1e2;
  hyp.cov = [lengthscales; 0.5*log(var(y0,1))];
  hyp.lik = 0.5*log(var(y0,1)/4);

  hyp_learned = minimize(hyp,@gp,-max_fevals,...
    @infExact, [], {@covSEard}, @likGauss, x, y0);

	logger.(tstamp).hyp_learned = hyp_learned;
	logger.(tstamp).training_time = toc(tstart);
	tstart = tic;
  [~,~,fmean,~,logpred] = gp(hyp_learned,@infExact,[],{@covSEard},...
      @likGauss,x,y0,xt,yt-mean(y));
  logger.(tstamp).prediction_time = toc(tstart);
  if zeromean
    fmean = fmean + mean(y);
  end      
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

