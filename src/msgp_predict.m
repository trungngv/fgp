function [smse,nlpd,mae,sqdiff,fmean,fvar,logpred,pred_time,valid]...
  = msgp_predict(num_inducing,x,xt,y,yt,zero_mean,W,train_label,test_label,display)
%MSGP_PREDICT [smse,nlpd,mae,sqdiff,fmean,fvar,logpred,pred_time,valid]...
%  = msgp_predict(x,xt,y,yt,zero_mean,W,train_label,test_label,display)
%
%   Prediction in MSGP model.
% 
% INPUT:
%   - num_inducing : number of inducing points per expert
%   - x, xt : training and test inputs
%   - y, yt : training and test outputs (for perf. measurements) 
%   - W : each column of W contains hyperparameters of an expert
%   - zero_mean : if training outputs transformed to have zero mean
%   - train_label : assignments of training data to experts
%   - test_label : (optional) assignments of test data to experts. if test_label is
%   empty, prediction of a point is by the most confidennt expert (i.e. the
%   expert with lowest predictive variance)
%   - display : true to print out smse, nlpd, mae, and sqdiff
% 
% OUTPUT:
%   - smse, nlpd, mae, sqdiff : average performance measures (standardised square error,
%      negative log predictive density, mean absolute error, sqdiff)
%   - fmean : predictive mean
%   - fvar : predictive var
%   - logpred : log predictive density
%   - pred_time : prediction time
%   - valid : false if any expert has more inducing points than training points (rarely occur)
%
% Trung V. Nguyen
% 06/2013
tstart = tic;
[Nt, dim] = size(xt); K = size(W,2);
valid = true;
if ~isempty(test_label)
  fmean = zeros(Nt,1); fvar = zeros(Nt,1); logpred = zeros(Nt,1);
else
  fmean = zeros(Nt,K); fvar = zeros(Nt,K); logpred = zeros(Nt,K);
end

for k=1:K
  indk = train_label == k;
  wk = W(:,k);
  xu = reshape(wk(1:end-dim-2),num_inducing,dim);
  hypk = fitc_hyp_to_gpml(wk,dim);
  try
    if ~isempty(test_label)
      test_indk = test_label == k;
      [fmean(test_indk),~,logpred(test_indk)] = gpmlFITC(x(indk,:),...
        y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,zero_mean);
    else
      [fmean(:,k),fvar(:,k),logpred(:,k)] = gpmlFITC(x(indk,:),...
        y(indk),xt,yt,xu,hypk,zero_mean);
    end
  catch eee
    valid = false;
  end
end

% predict based on confidence of experts
if isempty(test_label)
  [~,test_label] = min(fvar,[],2);
  linear_ind = sub2ind(size(fmean),(1:Nt)',test_label);
  fmean = fmean(linear_ind);
  fvar = fvar(linear_ind);
  logpred = logpred(linear_ind);
end

pred_time = toc(tstart);
smse = mysmse(yt,fmean,mean(y));
nlpd = -mean(logpred);
mae = mean(abs(fmean-yt));
sqdiff = sqrt(mean((fmean-yt).^2));
if display
  fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
  fprintf('nlpd = %.4f\n', -mean(logpred));
  fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
  fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2)));
end

