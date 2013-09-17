function logL = l2bound(y, Knm, Kmm, Lmm, diagKtilde, betaval, jitter)
% bound of Titsias'
% l2bound = l_{sor} - 0.5*betaval*trace(Knn - Knm*KmmInv*Kmn)
% betaval = 1/(sigman^2)
% Lmm: Cholesky factor of Kmm: Kmm = Lmm*Lmm'
% diagKtilde: vector: diag(Ktilde) = diag(Knn - Knm*KmmInv*Kmn)

lsor = margl_sor(y, Knm, Kmm, Lmm, betaval, jitter);


logL = lsor - 0.5*betaval*sum(diagKtilde);

return;




