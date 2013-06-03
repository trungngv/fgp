function [fw,dfw] = indpar_marginal(w,x,y,K,nu,func_assign)
%INDPAR_MARGINAL [fw,dfw] = indpar_marginal(w,x,y,K,nu,func_assign)
% 
% Negative log marginal and its derivative of the inducing partition model.
%
% 15/03/13
% Trung V. Nguyen
%
dim = size(x,2);
% unbox the parameters
% W = [w1 ... wK] where wk are parameters of the k-th partition
W = reshape(w,numel(w)/K,K);

% assign points to partition based on the basis points
%xxx= load('temp.mat');
label = feval(func_assign,W(1:end-dim-2,:),x,nu);
%save('temp.mat','label')
%fprintf('point changes = %d\n', sum(xxx.label ~= label));

% objective and derivatives 
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
