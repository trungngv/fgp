clear all; clc; close all;

%% Paths and general settings
%path(path(), '~/Dropbox/Matlab/utils');
%path(path(), '~/Dropbox/Matlab/gpml-matlab/gpml');
%path(genpath('~/Dropbox/Matlab/Netlab'), path());
covfunc   = 'covSEard';
PTRAIN    = 0.8; % Proportion of data-points for training
PIND      = 0.2; % 0.2 Proportion inducing points        
KMEANS    = 0;   % Use K-means for inducing locations
SIGMA2N   = 1e-7;
BETAVAL   = 1/SIGMA2N;

%% Learning configuration
cf.maxiter   = 100;
cf.tol       = 1e-3;
cf.lrate     = 0.01;
cf.Sinv0     = [];
cf.m0        = [];
cf.nbatch    = 2; % batch size 
cf.jitter   = 1e-7;

%% Generate samples from a GP
linfunc = @(x) sin(x) + cos(x) + cf.jitter;
x         = linspace(-10,10)';
D         = size(x,2);
nhyper    = eval(feval(covfunc));
loghyper  = log(ones(nhyper,1));
%f         = sample_gp(x, covfunc, loghyper, cf.jitter);
f	  = linfunc(x);
y         = f + sqrt(SIGMA2N)*randn(size(f)); % adds isotropic gaussian noise

%% Training, test 
Nall      = size(x,1); 
N         = ceil(PTRAIN*Nall);
idx       = randperm(Nall);
idx_train = idx(1:N);
idx_test  = idx(N+1:Nall);
xtrain    = x(idx_train,:); ytrain = y(idx_train); 
xtest     = x(idx_test,:);  ytest  = y(idx_test);

%% Inducing point locations
M     = ceil(PIND*N);
idx_z = randperm(N);
idx_z = idx_z(1:M);
z   = xtrain(idx_z,:); 
if (KMEANS)
    z   = kmeans(z, xtrain, foptions());
end

%% DELETE ME
% This is the optimal case, but inefficient
%cf.nbatch = N;
%cf.lrate  = 1;

%% Get data structures for computations of marginal likelihoods
Kmm             = feval(covfunc, loghyper, z);
%[diagKnn Kmn] = feval(covfunc, loghyper, z, xtrain);
%Knm = Kmn'; clear Kmn;
diagKnn         = feval(covfunc, loghyper, xtrain, 'diag'); % new api
Knm             = feval(covfunc, loghyper, xtrain, z); % new api
%Knm             = Kmn'; clear Kmn;
Lmm             = jit_chol(Kmm, 5);
Kmminv          = invChol(Lmm);
valK            = diagProd(Knm,Kmminv*Knm'); % diag(Knm*Kmminv*Kmn)
diagKtilde      = diagKnn - valK; % diag(Knn - Knm*KmmInv*Kmn)


%% computes the bounds
lsor = margl_sor(ytrain, Knm, Kmm, Lmm, BETAVAL, cf.jitter);
l2 = l2bound(ytrain, Knm, Kmm, Lmm, diagKtilde, BETAVAL, cf.jitter);


% R = 10;
% l3 = zeros(R,1);
% S = 1e-7*eye(M);
% for r = 1 : R
%     idx   = randperm(N);
%     idx   = idx(1:M);
%     m     = ytrain(idx);
%     l3(r) = l3bound(ytrain, Knm, Lmm, Kmminv, diagKtilde, BETAVAL, m, S);
% end
% %m = ytrain(idx_z); % TO DO PREDICTIONs at TZ instead
% %l3(R) = l3bound(ytrain, Knm, Lmm, Kmminv, diagKtilde, BETAVAL, m, S);
% figure;
% plot((1:R)', lsor*ones(R,1), 'r-'); hold on;
% plot((1:R)', l2*ones(R,1), 'g-'); hold on;
% plot((1:R)', l3, 'b-');
% legend({'L-SOR', 'L2', 'L3'});



%% learning variational parameters
%figure;
%semilogy((1:cf.maxiter)', lsor*ones(cf.maxiter,1), 'b-'); hold on;
%semilogy((1:cf.maxiter)', l2*ones(cf.maxiter,1), 'g-'); hold on;
%legend({'L-SOR', 'L2'});
[m S] = learn_q_gpsvi(ytrain, Knm, Lmm, Kmminv, diagKtilde, BETAVAL, cf);
%title('L3 vs L2 and LSOR');
%ylabel('ELBO');

%% predictions using gpsvi
[mupred varpred] = predict_gpsvi(Kmminv, covfunc, loghyper, m, S, z, x);

%% Plot training and predictions
%figure;
%[x idx]  = sort(x);
%mupred   = mupred(idx);
%varpred  = varpred(idx);
%plot_confidence_interval(x,mupred,sqrt(varpred),1);
%hold on;
%plot(x, mupred, 'r--'); 
%plot(x,f); hold on;
%plot(xtrain, ytrain, 'ro', 'MarkerSize', 8); hold on;
%plot(z, min(y)*ones(size(z,1), 1), 'kx', 'MarkerSize', 10); 
%ylabel('Predictive distribution');












































