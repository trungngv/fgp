function fw = spgp_lik_f(w,ind,n,del)

% spgp_lik_3: neg. log likelihood only for SPGP. Gaussian covariance with one
% lengthscale per dimension.
%
% ind -- indice of training targets and inputs
% n -- number of pseudo-inputs
% w -- parameters = [reshape(xb,n*dim,1); hyp]
%
%      hyp -- hyperparameters (dim+2 x 1)
%      hyp(1:dim) = log( b )
%      hyp(dim+1) = log( c )
%      hyp(dim+2) = log( sig )
%
%      where cov = c * exp[-0.5 * sum_d b_d * (x_d - x'_d)^2] 
%                       + sig * delta(x,x')
%
%      xb -- pseudo-inputs (n*dim)
%
% del -- OPTIONAL jitter (default 1e-6)
%
% fw -- likelihood
% dfw -- OPTIONAL gradients
%
% Edward Snelson (2006)
% Trung Nguyen (2013)
disp('f only')
global globalx; global globaly;
x = globalx(ind,:);
y = globaly(ind);
if nargin < 5; del = 1e-6; end % default jitter

[N,dim] = size(x); xb = reshape(w(1:end-dim-2),n,dim);
b = exp(w(end-dim-1:end-2)); c = exp(w(end-1)); sig = exp(w(end));

xb = xb.*repmat(sqrt(b)',n,1);
x = x.*repmat(sqrt(b)',N,1);

Q = xb*xb';
Q = repmat(diag(Q),1,n) + repmat(diag(Q)',n,1) - 2*Q;
Q = c*exp(-0.5*Q) + del*eye(n);

K = -2*xb*x' + repmat(sum(x.*x,2)',n,1) + repmat(sum(xb.*xb,2),1,N);
K = c*exp(-0.5*K);

L = chol(Q)';
V = L\K;
ep = 1 + (c-sum(V.^2)')/sig;
K = K./repmat(sqrt(ep)',n,1);
V = V./repmat(sqrt(ep)',n,1); y = y./sqrt(ep);
Lm = chol(sig*eye(n) + V*V')';
invLmV = Lm\V;
bet = invLmV*y;

% Likelihood
fw = sum(log(diag(Lm))) + (N-n)/2*log(sig) + ... 
      (y'*y - bet'*bet)/2/sig + sum(log(ep))/2 + 0.5*N*log(2*pi);
