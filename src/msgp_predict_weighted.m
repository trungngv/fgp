function [smse,nlpd,mae,sqdiff,fmean,fvar,logpred,pred_time,valid]...
  = msgp_predict_weighted(num_inducing,x,xt,y,yt,zero_mean,W,train_label,~,display)
%MSGP_PREDICT [smse,nlpd,mae,sqdiff,fmean,fvar,logpred,pred_time,valid]...
%  = msgp_predict(x,xt,y,yt,zero_mean,W,train_label,test_label,display)
%
%   Weighted prediction in MSGP model.
% 
% INPUT:
%   - num_inducing : number of inducing points per expert
%   - x, xt : training and test inputs
%   - y, yt : training and test outputs (for perf. measurements) 
%   - W : each column of W contains hyperparameters of an expert
%   - zero_mean : if training outputs transformed to have zero mean
%   - train_label : assignments of training data to experts
%   - test_label : []
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
test_label = [];
fmean = zeros(Nt,K); fvar = zeros(Nt,K); logpred = zeros(Nt,K);

for k=1:K
  indk = train_label == k;
  wk = W(:,k);
  xu = reshape(wk(1:end-dim-2),num_inducing,dim);
  hypk = fitc_hyp_to_gpml(wk,dim);
  try
    [fmean(:,k),fvar(:,k),logpred(:,k)] = gpmlFITC(x(indk,:),...
        y(indk),xt,yt,xu,hypk,zero_mean);
  catch eee
    valid = false;
  end
end

[~,~,dist,detD] = zhat_mahala(W(1:end-dim-2,:),xt,num_inducing);
p = exp(-0.5*dist) * ((2*pi)^(-dim/2)/sqrt(detD));
sum_p = sum(p,2);
p = p ./ repmat(sum_p,1,K);
fmean = sum(p.*fmean, 2);
logpred = log(sum(exp(logpred).*p,2));

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

