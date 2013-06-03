function [fw,dfw] = indpar_marginal_fast(w,K,nu,func_assign)
%INDPAR_MARGINAL_FAST [fw,dfw] = indpar_marginal_fast(w,K,nu,func_assign)
% 
% Negative log marginal and its derivative of the inducing partition model.
%
% 15/03/13
% Trung V. Nguyen
%
global globalx;
dim = size(globalx,2);
% unbox the parameters
% W = [w1 ... wK] where wk are parameters of the k-th partition
W = reshape(w,numel(w)/K,K);

% assign points to partition based on the basis points
%xxx= load('temp.mat');
label = feval(func_assign,W(1:end-dim-2,:),globalx,nu);
%save('temp.mat','label')
%fprintf('point changes = %d\n', sum(xxx.label ~= label));

% objective and derivatives 
tic;
fw = 0;
dfw = zeros(size(w,1),1);
pars = cell(K,1);
for k = 1:K
  indk = find(label==k); % size of each indk is much smaller than the full logical
  pars{k} = {size(globalx,1),W(:,k),indk,nu};
end

%multicore setting: can be passed in to avoid repetition
fEvalTimeSingle = 10; % default: 0.5
% use K / numcores
settings.nrOfEvalsAtOnce = 5; % default: 4
% see multicoredemo for explaination
settings.maxEvalTimeSingle = fEvalTimeSingle * 1.5;
settings.masterIsWorker = 1;
if nargout > 1
  results = startmulticoremaster(@spgp_lik_both,pars,settings);
  last_ind = 1;
  for k=1:K
    dfw(last_ind:(last_ind + numel(results{k}.dfw) - 1)) = results{k}.dfw;
    last_ind = last_ind + numel(results{k}.dfw);
    fw = fw + results{k}.fw;
  end
else
  results = startmulticoremaster(@spgp_lik_f,pars,settings);
  for k=1:K
    fw = fw + results{k};
  end
end
rtime = toc;
fprintf(' %.2f(s)\n',rtime);

% tic;
% global globaly;
% fw=0; dfw = zeros(size(w,1),1); last_ind=1;
% for k = 1:K
%   indk = label==k;
%   if nargout > 1 % optional derivatives
%     [fk, dfk] = spgp_lik(W(:,k),globaly(indk),globalx(indk,:),nu);
%     dfw(last_ind:(last_ind + numel(dfk) - 1)) = dfk;
%     last_ind = last_ind + numel(dfk);
%   else
%     fk = spgp_lik(W(:,k),globaly(indk),globalx(indk,:),nu);
%   end
%   fw = fw + fk;
% end
% toc
% disp('------>by sequential')
