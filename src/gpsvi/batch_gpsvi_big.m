clear all; clc; 

%% Paths and general settings
path(path(), '~/projects/myutils');
path(path(), genpath('~/projects/fagpe'));
covfunc   = 'covSEard';
KMEANS    = 0;   % Use K-means for inducing locations
SIGMA2N   = 1e-7;
BETAVAL   = 1/SIGMA2N;


%% Learning configuration
cf.maxiter   = 1000;
cf.tol       = 1e-3;
cf.lrate     = 0.01;
cf.Sinv0     = [];
cf.m0        = [];
cf.nbatch    = 1000; % batch size 
cf.jitter   = 1e-7;
cf.covFunc = covfunc;

%% load data
dataset = 'fagpe/data/song100k';
[xtrain,ytrain,xtest,ytest] = load_data(dataset, 'song100k');
disp('finish reading data')

% load learned hyperparameters from subset of data
load('fagpe/output/song100k_sod.mat'); 
tstamps = fieldnames(logger);
loghyp_learned = zeros(size(xtrain,2)+1,5);
for i=1:numel(tstamps)
  loghyp_learned(:,i) = logger.(tstamps{i}).hyp_learned.cov;
end
clear logger;
output_file = 'fagpe/output/song100k_gpsvi.mat';
try
  load(output_file, 'logger');
catch eee
  logger = [];
  save(output_file, 'logger');
end

%
%% inducing points for training 
Nall      = size(xtrain,1); 
N = Nall;
y_mean = mean(ytrain);
ytrain = ytrain - y_mean;
learned_log_hyper = [];

theseed = 1110;
M = 2000;
for sss=1:5
seed = theseed+sss;
rng(seed,'twister');
strdate = datestr(now, 'mmmdd');
tstamp = [strdate 't' num2str(tic)];
logger.(tstamp).rng = seed;
logger.(tstamp).N_training = N;
logger.(tstamp).cf = cf;
logger.(tstamp).num_inducing = M;
idx_z = randperm(N);
idx_z = idx_z(1:M);
z   = xtrain(idx_z,:); 
if (KMEANS)
    z   = kmeans(z, xtrain, foptions());
end

%% Get data structures for computations of marginal likelihoods
tstart = tic;
loghyper = loghyp_learned(:,sss);
cf.loghyper = loghyper;
Kmm             = feval(covfunc, loghyper, z);
%[diagKnn Kmn] = feval(covfunc, loghyper, z, xtrain);
%Knm = Kmn'; clear Kmn;
%diagKnn         = feval(covfunc, loghyper, xtrain, 'diag'); % new api
%Knm             = feval(covfunc, loghyper, xtrain, z); % new api
%Knm             = Kmn'; clear Kmn;
Lmm             = jit_chol(Kmm, 5);
Kmminv          = invChol(Lmm);
%valK            = diagProd(Knm,Kmminv*Knm'); % diag(Knm*Kmminv*Kmn)
%diagKtilde      = diagKnn - valK; % diag(Knn - Knm*KmmInv*Kmn)

disp('learning...')
[m S] = learn_q_gpsvi_big(xtrain, ytrain, z, Lmm, Kmminv, BETAVAL, cf);
logger.(tstamp).training_time = toc(tstart);
logger.(tstamp).m = m;
logger.(tstamp).S = S;
tstart = tic;
disp('making prediction...')
[mupred varpred] = predict_gpsvi(Kmminv, covfunc, loghyper, m, S, z, xtest);
logger.(tstamp).prediction_time = toc(tstart);
mupred = mupred + y_mean;
logger.(tstamp).smse = mysmse(ytest,mupred,y_mean);
logger.(tstamp).nlpd = mynlpd(ytest,mupred,varpred);
logger.(tstamp).mae = mean(abs(mupred-ytest));
logger.(tstamp).sqdiff = sqrt(mean((mupred-ytest).^2));
save(output_file, 'logger');
end

