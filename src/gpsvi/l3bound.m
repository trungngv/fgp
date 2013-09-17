function logL  = l3bound(y, Knm, Lmm, Kmminv, diagKtilde, betaval, m, S, jitter)
% diagKtilde: vector: diag(Ktilde) = diag(Knn - Knm*KmmInv*Kmn)

M = size(S,1);
N = size(Knm,1);

%%
%logN   = - sum( (betaval/2)*((y - Knm*Kmminv*m).^2) - (1/2)*log(2*pi*(1/betaval)) );
logN   = - (betaval/2)*sum((y - Knm*Kmminv*m).^2) - (N/2)*log(2*pi/betaval) ;


%%
ltilde = (0.5)*betaval*sum(diagKtilde );

%%
A      = Knm*Kmminv;
ltrace = (0.5)*betaval*sum( sum ( (S.*(A'*A)) ));

%%
%Ls           = chol_safe(S, jitter);
Ls = jit_chol(S);
logdetS      = logdetChol(Ls); 
logdetKmm    = logdetChol(Lmm); 
%lkl    = (1/2)*(sum(sum(Kmminv.*S))) + m'*Kmminv*m - logdet(S*Kmminv) - M;
lkl    = 0.5*( (sum(sum(Kmminv.*S))) + m'*Kmminv*m - logdetS + logdetKmm - M );


logL = logN - ltilde - ltrace - lkl;


return;


