function [fmean,fvar,logpred] = gpmlFITC(x,y,xt,yt,xu,hyp,flag)
%GPMLFITC [fmean,fvar,logpred] = gpmlFITC(x,y,xt,yt,xu,hyp,flag)
%   
%  Prediction with GP using FITC sparse approximation. Hyper-parameters and
%  inducing points are fixed (given). This method uses the GPML library.
%  
%  No data pre-processing for input is performed by this method. The
%  outputs are automatically zero-mean unless FLAG is set to false.
%
% Trung V. Nguyen
% 25/02/13

% set-up model
covfunc = {@covFITC,{@covSEard},xu};
likfunc = @likGauss;
if nargin == 6 || flag
  y0 = y - mean(y); % zero-mean
  yt0 = yt - mean(y); 
  [~,~,fmean,fvar,logpred] = gp(hyp,@infFITC,[],covfunc,likfunc,x,y0,xt,yt0);
  fmean = fmean + mean(y);
else
  [~,~,fmean,fvar,logpred] = gp(hyp,@infFITC,[],covfunc,likfunc,x,y,xt,yt);
end

