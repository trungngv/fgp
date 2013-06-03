function label = zmap_msgp(W,x,y,nu,train_label)
%ZMAP_MSGP label = zmap_msgp(W,x,y,nu,train_label)
%   
% Compute maximum a posteriori (MAP) of label assignment z_{MAP} using
% shared diagonal covariance matrix. 
%
% INPUT
%   - W : (nu*D + dim + 2) x K 
%   - x : N x D
%   - nu : num inducing
%   - train_label : label of training points
% OUTPUT
%   - label : new partition labels
%
% Trung Nguyen
% 05/2013
dim = size(x,2);
rho = compute_distance(W(1:end-dim-2,:),x,nu);
rho = rho + compute_llh(W,x,y,nu,train_label);
[~,label] = min(rho,[],2);
end

% first term in rho: cost O(NK)
function d = compute_distance(W,x,nu)
  K = size(W,2);  dim = size(x,2);
  centers = partition_centers(W,nu,K,dim);
  Mu = zeros(size(W));
  for k=1:K
    matMu = repmat(centers(k,:),nu,1); % nu x dim
    Mu(:,k) = matMu(:); % (nu*dim)x1 as W(:,k)
  end
  % sum over partitions and then reshape
  diagCov = reshape(sum((W-Mu).^2,2),nu,dim);
  % sum over xu and normalize (1xdim)
  diagCov = sum(diagCov)/(nu*K-K);
  diagCov = diagCov.^ 0.5;

  nx = (x ./ repmat(diagCov,size(x,1),1))';
  nmu = (centers ./ repmat(diagCov,K,1))';
  % (x-mu)^T Sigma^{-1} (x-mu) = (x-mu)^T Lambda (x-mu)
  %                            = (Lamba^(1/2)(x-mu))^T (Lamba^(1/2)(x-mu)) 
  % N x K distance from each point to partition centers
  d = sq_dist(nx,nmu);
end

% total cost : O(NKM^2) due to Luu'\Ku which compute all X wrt one u =
% O(NBM) < O(NB^2) in sparse GP
function llh = compute_llh(W,x,y,nu,label)
  K = size(W,2); dim = size(x,2);
  llh = zeros(size(x,1),K);
  for k=1:K
    indk = label == k; yk = y(indk,:); wk = W(:,k);
    % covariance matrix
    xu = reshape(wk(1:end-dim-2),nu,dim);
    cov = {@covFITC,{@covSEard},xu};
    hypk = fitc_hyp_to_gpml(wk,dim);
    % K(X,X), K(U,U), K(U,X) wrt expert k
    [diagK,Kuu,Ku] = feval(cov{:}, hypk.cov, x);
    sn2  = exp(2*hypk.lik);                              % noise variance of likGauss
    snu2 = 1e-6*sn2;                              % hard coded inducing inputs noise
    Luu  = chol(Kuu + snu2*eye(nu));                       % Kuu + snu2*I = Luu'*Luu
    V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q
    % total cost for lambda: O(KNM^2) due to (Luu')Ku = O(M^3) + O(NM^2)
    lambda = diagK + sn2 - sum(V.*V,1)';      % D + sn2*eye(n) = diag(K) + sn2 - diag(Q)
    % K(Uk,Xk) Lambda^{-1} K(Xk,Uk) = A inv(D) A = (AL)(AL))'
    % where L = D.^{-0.5} = 1 / sqrt(D)
    AL = Ku(:,indk).*repmat(1./sqrt(lambda(indk)'),nu,1);
    Psik = Kuu + AL*AL';
    % K(X,Uk) inv(Psik) K(Uk,Xk) (inv(lambda(xk))*yk)
    m = Ku'*(Psik\(Ku(:,indk)*(yk./lambda(indk)))); % O(N*M+M^3+NM/K)
    llh(:,k) = ((y - m).^2)./lambda + log(lambda);
  end
end

% second (diagonal) term in rho
% cost O(KNM^2) due to (Luu')Ku = O(M^3) + O(NM^2)
function lambda = compute_lambda(W,x,nu)
  K = size(W,2); dim = size(x,2);
  lambda = zeros(size(x,1),K);
  for k=1:K
    wk = W(:,k);
    xu = reshape(wk(1:end-dim-2),nu,dim);
    cov = {@covFITC,{@covSEard},xu};
    hypk = fitc_hyp_to_gpml(wk,dim);
    % K(X,X), K(U,U), K(U,X) wrt expert k
    [diagK,Kuu,Ku] = feval(cov{:}, hypk.cov, x);
    sn2  = exp(2*hypk.lik);                              % noise variance of likGauss
    snu2 = 1e-6*sn2;                              % hard coded inducing inputs noise
    Luu  = chol(Kuu + snu2*eye(nu));                       % Kuu + snu2*I = Luu'*Luu
    V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q
    lambda(:,k) = diagK + sn2 - sum(V.*V,1)';      % D + sn2*eye(n) = diag(K) + sn2 - diag(Q)
  end
end
