function [w,fval,zmap] = msgp_train_map(w,x,y,K,nu,max_iters,hyp_iters)
%MSGP_MAP_TRAIN [w,fval,zmap] = msgp_train_map(w,x,y,K,nu,max_iters,hyp_iters)
% 
% Training for mixture of sparse gaussian process.
% This version uses the true zmap for assignments (equation 17 and 13).
%
% 29/05/13
% Trung V. Nguyen
%
fval = [];
dim = size(x,2);
delta = 1e-5;
W = reshape(w,numel(w)/K,K);
zmap = zhat_mahala(W(1:end-dim-2,:),x,nu); % init
for iter=1:max_iters
  % W = [w1 ... wK] where wk are parameters of the k-th partition
  W = reshape(w,numel(w)/K,K);
  % e-step: find zmap and accept only if increases objective
  newzmap = zmap_msgp(W,x,y,nu,zmap);
  newobj = msgp_marginal(w,x,y,K,nu,newzmap);
  if isempty(fval) || newobj < fval(end)
    zmap = newzmap;
  end
  [w,lastfval] = minimize(w,@msgp_marginal,hyp_iters,x,y,K,nu,zmap);
  if isempty(fval),       fval = lastfval;
  else    % convergence check
    if abs(fval(end) - lastfval(end)) < delta
      break;
    end
    fval = [fval; lastfval];
  end  
end

function [fw, dfw] = msgp_marginal(w,x,y,K,nu,label)
% objective and derivatives 
W = reshape(w,numel(w)/K,K);
fw = 0;
dfw = zeros(size(w,1),1);
last_ind = 1;
for k = 1:K
  indk = label==k;
  if nargout > 1 % optional derivatives
    [fk, dfk] = spgp_lik(W(:,k),y(indk),x(indk,:),nu);
    dfw(last_ind:(last_ind + numel(dfk) - 1)) = dfk;
    last_ind = last_ind + numel(dfk);
  else
    fk = spgp_lik(W(:,k),y(indk),x(indk,:),nu);
  end
  fw = fw + fk;
end
