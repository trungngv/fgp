function [mu,s2,t_train,t_test,yb] = spgp_pred(y,x,xb,xt,hyp,del)

% spgp_pred_2: predictive distribution for SPGP given hyperparameters
% and pseudo-inputs
%
% y -- training targets (N x 1)
% x -- training inputs (N x dim)
% xb -- pseudo inputs (n x dim)
% xt -- test inputs (Nt x dim)
% hyp -- hyperparameters (including noise)
%        for Gaussian covariance: (dim+2 x 1)
% 
%       hyp(1:dim) = log( b )
%       hyp(dim+1) = log( c )
%       hyp(dim+2) = log( sig )
%
%       where cov = c * exp[-0.5 * sum_d b_d * (x_d - x'_d)^2] 
%                       + sig * delta(x,x')
% 
% del -- OPTIONAL jitter (default 1e-6)
%
% mu -- predictive mean
% s2 -- predictive variance of latent function
%       (add noise if full variance is required)
% 
% t_train, t_test -- OPTIONAL separate training and prediction times 
% yb -- OPTIONAL posterior mean pseudo targets
%
% Edward Snelson (2006)

if nargin < 6; del = 1e-6; end % default jitter

[N,dim] = size(x); n = size(xb,1); Nt = size(xt,1);
sig = exp(hyp(end)); % noise variance

% precomputations
tic;
K = kern(xb,xb,hyp) + del*eye(n);
L = chol(K)';
K = kern(xb,x,hyp);
V = L\K;
ep = 1 + (kdiag(x,hyp)-sum(V.^2,1)')/sig;
V = V./repmat(sqrt(ep)',n,1); y = y./sqrt(ep);
Lm = chol(sig*eye(n) + V*V')';
bet = Lm\(V*y);
clear V
t_train = toc;

% test predictions
tic;
K = kern(xb,xt,hyp);
lst = L\K;
clear K
lmst = Lm\lst;
mu = (bet'*lmst)';

s2 = kdiag(xt,hyp) - sum(lst.^2,1)' + sig*sum(lmst.^2,1)';
t_test = toc;


% OPTIONAL posterior mean pseudo targets
if nargout > 4; 
  yb = L*(Lm'\bet);
end


% Can replace with whatever kernel function you want:
function K = kern(x1,x2,hyp)

[n1,dim] = size(x1); n2 = size(x2,1);
b = exp(hyp(1:end-2)); c = exp(hyp(end-1));

x1 = x1.*repmat(sqrt(b)',n1,1);
x2 = x2.*repmat(sqrt(b)',n2,1);

K = -2*x1*x2' + repmat(sum(x2.*x2,2)',n1,1) + repmat(sum(x1.*x1,2),1,n2);
K = c*exp(-0.5*K);
% K = c*gauss_Knm(x1,x2,sqrt(2));

function Kd = kdiag(x,hyp);

c = exp(hyp(end-1));
Kd = repmat(c,size(x,1),1);