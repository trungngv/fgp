% demo of inducing (basis) based partition
%clear; close all;

%func_assign = 'assign_nearest_center';
func_assign = 'zhat_mahala';
datasets_dir = 'fagpe/data/';
dataset = 'song100k';

[x,y,xt,yt] = load_data([datasets_dir dataset], dataset);
y0 = y-mean(y);
disp('finished reading data')

dim = size(x,2);
K = 20;
nu = 300;
w = results.optim_w;
W = reshape(w,numel(w)/K,K);
label = feval(func_assign,W(1:end-dim-2,:),x,nu);
[test_label,~,posterior] = feval(func_assign,W(1:end-dim-2,:),xt,nu);
posterior = exp(-0.5*posterior) + 1e-15;
posterior = posterior ./ repmat(sum(posterior,2),1,K);

Nt = size(xt,1);
valid = true;
fmean = zeros(Nt,K); logpred = zeros(Nt,K);
tstart = tic;
for k=1:K
  indk = label == k;
  wk = W(:,k);
  xu = reshape(wk(1:end-dim-2),nu,dim);
  hypk = fitc_hyp_to_gpml(wk,dim);
  try
    [fmean(:,k),~,logpred(:,k)] = gpmlFITC(x(indk,:),...
      y(indk),xt,yt,xu,hypk,true);
  catch eee
    valid = false;
    fmean(:,k) = repmat(mean(y(indk)),size(yt,1),1);
  end
end
% the weighting (posterior)
fmean = sum(fmean .* posterior, 2);
logpred = sum(logpred .* posterior, 2);

% % predict old
% Nt = size(xt,1);
% valid = true;
% fmean = zeros(Nt,1); logpred = zeros(Nt,1);
% tstart = tic;
% for k=1:K
%   indk = label == k;
%   test_indk = test_label == k;
%   wk = W(:,k);
%   %TODO: use a unboxing function
%   xu = reshape(wk(1:end-dim-2),nu,dim);
%   hypk = fitc_hyp_to_gpml(wk,dim);
%   try
%     [fmean(test_indk),~,logpred(test_indk)] = gpmlFITC(x(indk,:),...
%       y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,true);
%   catch eee
%     valid = false;
%     fmean(test_indk) = repmat(mean(y(indk)),sum(test_indk),1);
%   end
% end

fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2))); 
fprintf('nlpd = %.4f\n', -mean(logpred));
figure; hist(abs(fmean-yt),200);

