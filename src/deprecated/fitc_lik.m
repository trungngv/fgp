function [fw,dfw] = fitc_lik(w,y,x,n,del)
%FITC_LIK [fw,dfw] = fitc_lik(w,y,x,n,del)
% spgp_lik_3: neg. log likelihood for SPGP and gradients with respect to
% pseudo-inputs and hyperparameters. Gaussian covariance with one
% lengthscale per dimension.
% 
% modified by TVN to use the same hyperparameter representation
% as the GPML framework
% TODO: must modify the gradient as well
%
% y -- training targets (N x 1)
% x -- training inputs (N x dim)
% n -- number of pseudo-inputs
% w -- parameters = [reshape(xb,n*dim,1); hyp]
%
%      hyp -- hyperparameters (dim+2 x 1)
%      hyp(1:dim) = log(sigma_f)
%      hyp(dim+1) = log(ell_d)
%      hyp(dim+2) = log(sigma_n) as in gpml
%
%      where cov = \sigma_f^2 * exp[-0.5 * sum_d ((x_d - x'_d)/ell)^2] 
%                       + \sigma_n^2 * delta(x,x')
%
%      xb -- pseudo-inputs (n*dim)
%
% del -- OPTIONAL jitter (default 1e-6)
%
% fw -- likelihood
% dfw -- OPTIONAL gradients
%
% Edward Snelson (2006)
if nargin < 5; del = 1e-6; end % default jitter

[N,dim] = size(x); xb = reshape(w(1:end-dim-2),n,dim);
ell2 = exp(-2*w(end-dim-1:end-2)); sf2 = exp(2*w(end-1)); sn2 = exp(2*w(end));

xb = xb.*repmat(sqrt(ell2)',n,1);
x = x.*repmat(sqrt(ell2)',N,1);

Q = xb*xb';
Q = repmat(diag(Q),1,n) + repmat(diag(Q)',n,1) - 2*Q;
Q = sf2*exp(-0.5*Q) + del*eye(n);

K = -2*xb*x' + repmat(sum(x.*x,2)',n,1) + repmat(sum(xb.*xb,2),1,N);
K = sf2*exp(-0.5*K);

L = chol(Q)';
V = L\K;
ep = 1 + (sf2-sum(V.^2)')/sn2;
K = K./repmat(sqrt(ep)',n,1);
V = V./repmat(sqrt(ep)',n,1); y = y./sqrt(ep);
Lm = chol(sn2*eye(n) + V*V')';
invLmV = Lm\V;
bet = invLmV*y;

% Likelihood
fw = sum(log(diag(Lm))) + (N-n)/2*log(sn2) + ... 
      (y'*y - bet'*bet)/2/sn2 + sum(log(ep))/2 + 0.5*N*log(2*pi);

% OPTIONAL derivatives
if nargout > 1

% precomputations
Lt = L*Lm;
B1 = Lt'\(invLmV);
b1 = Lt'\bet;
invLV = L'\V;
invL = inv(L); invQ = invL'*invL; clear invL
invLt = inv(Lt); invA = invLt'*invLt; clear invLt
mu = ((Lm'\bet)'*V)';
sumVsq = sum(V.^2)'; clear V
bigsum = y.*(bet'*invLmV)'/sn2 - sum(invLmV.*invLmV)'/2 - (y.^2+mu.^2)/2/sn2 ...
         + 0.5;
TT = invLV*(invLV'.*repmat(bigsum,1,n));

% pseudo inputs and lengthscales
for i = 1:dim
% dnnQ = (repmat(xb(:,i),1,n)-repmat(xb(:,i)',n,1)).*Q;
% dNnK = (repmat(x(:,i)',n,1)-repmat(xb(:,i),1,N)).*K;
dnnQ = snelson_dist(xb(:,i),xb(:,i)).*Q;
dNnK = snelson_dist(-xb(:,i),-x(:,i)).*K;

epdot = -2/sn2*dNnK.*invLV; epPmod = -sum(epdot)';

dfxb(:,i) = - b1.*(dNnK*(y-mu)/sn2 + dnnQ*b1) ...
    + sum((invQ - invA*sn2).*dnnQ,2) ...
    + epdot*bigsum - 2/sn2*sum(dnnQ.*TT,2); 

dfb(i,1) = (((y-mu)'.*(b1'*dNnK))/sn2 ...
           + (epPmod.*bigsum)')*x(:,i);

dNnK = dNnK.*B1; % overwrite dNnK
dfxb(:,i) = dfxb(:,i) + sum(dNnK,2);
dfb(i,1) = dfb(i,1) - sum(dNnK,1)*x(:,i);

dfxb(:,i) = dfxb(:,i)*sqrt(ell2(i));

dfb(i,1) = dfb(i,1)/sqrt(ell2(i));
dfb(i,1) = dfb(i,1) + dfxb(:,i)'*xb(:,i)/ell2(i);
dfb(i,1) = dfb(i,1)*sqrt(ell2(i))/2;
end

% size
epc = (sf2./ep - sumVsq - del*sum((invLV).^2)')/sn2;

dfc = (n + del*trace(invQ-sn2*invA) ... 
     - sn2*sum(sum(invA.*Q')))/2 ...
    - mu'*(y-mu)/sn2 + b1'*(Q-del*eye(n))*b1/2 ... 
      + epc'*bigsum;

% noise
dfsig = sum(bigsum./ep);

dfw = [reshape(dfxb,n*dim,1);dfb;dfc;dfsig];

end