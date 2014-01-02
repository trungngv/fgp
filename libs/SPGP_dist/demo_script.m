clear

% load demo data set (1D inputs for easy visualization -
% this script should work fine for multidimensional inputs)

x = load('train_inputs');
y = load('train_outputs');
xtest = load('test_inputs');
me_y = mean(y); y0 = y - me_y; % zero mean the data

[N,dim] = size(x);

M = 20; % number of pseudo-inputs

% initialize pseudo-inputs to a random subset of training inputs
[dum,I] = sort(rand(N,1)); clear dum;
I = I(1:M);
xb_init = x(I,:);

% initialize hyperparameters sensibly (see spgp_lik for how
% the hyperparameters are encoded)
hyp_init(1:dim,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
hyp_init(dim+1,1) = log(var(y0,1)); % log size 
hyp_init(dim+2,1) = log(var(y0,1)/4); % log noise

% optimize hyperparameters and pseudo-inputs
w_init = [reshape(xb_init,M*dim,1);hyp_init];
[w,f] = minimize(w_init,'spgp_lik',-200,y0,x,M);
% [w,f] = lbfgs(w_init,'spgp_lik',200,10,y0,x,M); % an alternative
xb = reshape(w(1:M*dim,1),M,dim);
hyp = w(M*dim+1:end,1);


% PREDICTION
[mu0,s2] = spgp_pred(y0,x,xb,xtest,hyp);
mu = mu0 + me_y; % add the mean back on
% if you want predictive variances to include noise variance add noise:
s2 = s2 + exp(hyp(end));


%%%%%%%%%
% Plotting - just for 1D demo - remove for real data set
% Hopefully, the predictions should look reasonable

clf
hold on
plot(x,y,'.m') % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(xb,-2.75*ones(size(xb)),'k+','markersize',20)
hold off
axis([-3 10 -3 2])