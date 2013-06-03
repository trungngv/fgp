function [fmean,fvar,logpred,valid] = gps_fitc_predict(models,x,y,xt,yt,label,...
  test_label,zeromean)
%GPS_FITC_PREDICT [fmean,fvar,logpred] = gps_fitc_predict(models,x,y,xt,yt,label,test_label)
%   Prediction for all partitions.
%
% 
K = numel(models);
Nt = size(xt,1);
valid = true;
fmean = zeros(Nt,1); fvar = zeros(Nt,1); logpred = zeros(Nt,1);
for k=1:K
  indk = label == k;
  test_indk = test_label == k;
  if ~isempty(models{k})
    [fmean(test_indk),fvar(test_indk),logpred(test_indk)] = gpmlFITC(x(indk,:),...
      y(indk),xt(test_indk,:),yt(test_indk),models{k}.xu,models{k}.hyp,zeromean);
  else
    valid = false;
    fmean(test_indk) = repmat(mean(y),sum(test_indk),1);
  end
end
