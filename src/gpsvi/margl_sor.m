function lsor = margl_sor(y, Knm, Kmm, Lmm, betaval, jitter)
% Marginal likelihood for SOR approximation
% Equation taken from my research notes (dropbox)
% betaval = 1/(sigman^2)
% Lmm: Cholesky factor of Kmm: Kmm = Lmm*Lmm'

N = size(Knm,1);

%% First log determinant term
C     = Knm'*Knm + (1/betaval)*Kmm;
%Lc    = chol_safe(C, jitter);
Lc = jit_chol(C);
%ldetc = logdetChol(Lc);
ldetc = 2*sum(log(diag(Lc)));

%% second log determinant term
%ldetKmm = logdetChol(Lmm); % 2*sum(log(diag(Lmm)));
ldetKmm = 2*sum(log(diag(Lmm)));

%% first quatratic germ
lquad1 = betaval*(y'*y);

%% Second quadratic term
m      = Knm'*y;
alpha  = solve_chol(Lc',m);
lquad2 = betaval*m'*alpha;

lsor = -0.5*( ldetc - ldetKmm + lquad1 - lquad2 + N*log(2*pi) );




return;




