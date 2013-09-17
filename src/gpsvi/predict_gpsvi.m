function [mu var] = predict_gpsvi(Kmminv, covfunc, loghyper, m, S, z, xstar)
% Makes predictions with gpsvi
% z are the inducing point lcoations
% xstar are the test-pints
if size(xstar,1) > 1000
  nbatch = 10;
else
  nbatch = 1;
end
bsize = size(xstar,1)/nbatch;
mu = [];
var= [];
for i=1:nbatch
  %[Kss Kms] = feval(covfunc, loghyper, z, xstar);
  %Ksm       = Kms';
  pos = (i-1)*bsize + 1;
  if i < nbatch
    bxstar = xstar(pos:i*bsize,:);
  else
    bxstar = xstar(pos:end,:);
  end
  Kss = feval(covfunc, loghyper, bxstar, 'diag');
  Kms = feval(covfunc, loghyper, z, bxstar);
  Ksm = Kms';

  %% Mean of predictive distribution
  %mu          = Ksm*(Kmminv*m);
  mu = [mu; Ksm*(Kmminv*m)];

  %% variance of predictive distribution
  % we can also compute full covariance at a higher cost
  % diag(Ksm * kmminv * S * Kmmonv *Kms) 
  var_1 =  sum(Kms.*(Kmminv*S*Kmminv*Kms),1)';
  var_2 =  sum(Kms.*(Kmminv*Kms),1)';   
  %var = var_1 + Kss - var_2;
  var = [var; var_1 + Kss - var_2];

end

return;

