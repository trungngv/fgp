function [fmean,logpred,valid] = prob_indpar_predict(x,y,xt,yt,W,num_inducing,numz,numzstar)
%PROB_INDPAR_PREDICT [fmean,logpred,valid] = prob_indpar_predict(x,y,xt,yt,W,num_inducing)
%
% Prediction
%
% Usage:
%    [fmean,logpred] = prob_indpar_predict(x,y,xt,yt,W,num_inducing)
%       get predictive mean and log predictive 
%
% Trung Nguyen
% 15/04/13
disp('making prediction')
dim = size(x,2);
K = size(W,2);
Z = sample_z(numz,W(1:end-dim-2,:),x,num_inducing);
Zstar = sample_z(numzstar,W(1:end-dim-2,:),xt,num_inducing);
Nt = size(xt,1);
fmean_all = zeros(Nt,1);
logpred_all = zeros(Nt,1);
fmean_s = zeros(Nt,1);    % one sample of Z and Z'
logpred_s = zeros(Nt,1);
valid = true;
for iz=1:numz    % prediction for one z  
  for izstar=1:numzstar
    for k=1:K
      indk = Z(iz,:) == k;
      test_indk = Zstar(izstar,:) == k;
      wk = W(:,k);
      xu = reshape(wk(1:end-dim-2),num_inducing,dim);
      hypk = fitc_hyp_to_gpml(wk,dim);
      try 
        [fmean_s(test_indk),~,logpred_s(test_indk)] = gpmlFITC(x(indk,:),y(indk),...
          xt(test_indk,:),yt(test_indk),xu,hypk,false);
      catch eee
        valid = false;
      end
      fmean_all(test_indk) = fmean_all(test_indk) + fmean_s(test_indk);
      logpred_all(test_indk) = logpred_all(test_indk) + logpred_s(test_indk);
    end
  end  
end
fmean = fmean_all/(numz*numzstar);
logpred = logpred_all/(numz*numzstar);
% var due to mixture of gaussian
%fvar = mean(fvars)+mean((fmeans-repmat(fmean,numz,1)).^2);
