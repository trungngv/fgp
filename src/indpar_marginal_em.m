function [fw,dfw] = indpar_marginal_em(w,x,y,K,M)
%INDPAR_MARGINAL_EM [fw,dfw] = indpar_marginal_em(w,x,y,K,M)
% 
% Negative log marginal and its derivative of the inducing partition model.
% This implementation labels of inducing inputs are unknown.
% 
% M : total number of inducing points
%
% 13/05/13
% Trung V. Nguyen
%
dim = size(x,2);
% unbox the parameters
% w = [u(:) theta(:)] 
% U = M x dim
U = reshape(w(1:M*dim),M,dim);
% Theta = [theta1,...,thetak]
Theta = reshape(w(M*dim+1:end),dim+2,K);

% assign points to partition based on the basis points
%xxx= load('temp.mat');
[ulabel,xlabel] = zmap_mahala_em(U,K,x,[]);
%save('temp.mat','label')
%fprintf('point changes = %d\n', sum(xxx.label ~= label));

% objective and derivatives 
fw = 0;
dfU = zeros(size(U));
dfTheta = zeros(size(Theta));
for k = 1:K
  indk_x = xlabel==k;
  indk_u = ulabel == k;
  Uk = U(indk_u,:);
  Nuk = size(Uk,1);
  if nargout > 1 % optional derivatives
    [fk, dfk] = spgp_lik([Uk(:);Theta(:,k)],y(indk_x),x(indk_x,:),Nuk);
    dfU(indk_u,:) = reshape(dfk(1:Nuk*dim),Nuk,dim);
    dfTheta(:,k) = dfk(Nuk*dim+1:end);
  else
    fk = spgp_lik([Uk(:),Theta(:,k)],y(indk_x),x(indk_x,:),size(Uk,1));
  end
  fw = fw + fk;
end
dfw = [dfU(:);dfTheta(:)];

