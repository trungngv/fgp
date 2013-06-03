function [fw,dfw] = prob_indpar_marginal(w,x,y,K,num_inducing,numz)
%PROB_INDPAR_MARGINAL [fw,dfw] = prob_indpar_marginal(w,x,y,K,num_inducing,numz)
% 
% Negative log marginal and its derivative of the inducing partition model.
% This function uses the probabilistic partition assignment fucntion.
%
% 08/04/13
% Trung V. Nguyen
%
dim = size(x,2);
% unbox the parameters
% W = [w1 ... wK] where wk are parameters of the k-th partition
W = reshape(w,numel(w)/K,K);
centers = partition_centers(W(1:end-dim-2,:),num_inducing,K,dim);
Z = sample_z(numz,W(1:end-dim-2,:),x,num_inducing,centers);
nlogpz = zeros(numz,1); % -log p(y|z,Xu,\theta)
% each column is - dlog p(y|z,Xu,\theta) / d{Xu,\theta}
ndlogpz = zeros(size(w,1),numz);
for iz=1:numz
  if nargout > 1
    [nlogpz(iz),ndlogpz(:,iz)] = single_marginal(W,numel(w),x,y,K,num_inducing,Z(iz,:));
  else  
    nlogpz(iz) = single_marginal(W,numel(w),x,y,K,num_inducing,Z(iz,:));
  end  
end
% nlog = -log(1/nz) - logsum(logpz)
%TODO: const factor from logpz?
fw = log(numz) - logsum(-nlogpz);
if nargout > 1
  dfw = zeros(size(w,1),1);
  for iz=1:numz
    dfw = dfw + ndlogpz(:,iz)*(exp(fw - nlogpz(iz)));
  end
  dfw = dfw / numz;
end

% objective and derivatives given a partition assignment
function [fw dfw] = single_marginal(W,numel_w,x,y,K,num_inducing,label)
fw = 0;
dfw = zeros(numel_w,1);
last_ind = 1;
for k = 1:K
  indk = label==k;
  if nargout > 1 % optional derivatives
    [fk, dfk] = spgp_lik(W(:,k),y(indk),x(indk,:),num_inducing);
    dfw(last_ind:(last_ind+numel(dfk)- 1)) = dfk;
    last_ind = last_ind + numel(dfk);
  else
    fk = spgp_lik(W(:,k),y(indk),x(indk,:),num_inducing);
  end
  fw = fw + fk;
end

