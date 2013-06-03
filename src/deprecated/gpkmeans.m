function [smse,msll,partition,hyp,nmll] = gpkmeans(K,x,y,xt,yt)
%GPKMEANS gpkmeans(x,y,xt,yt)
%   
% GP via k-means
%
% NOTE: there are currently 2 ways of transforming the output to be of
% 0-mean: globally or locally for each cluster. Depending on how it is
% transformed, the SMSE must be computed according to the transformation
% because it depends on mean of the outputs!
%
% Trung V. Nguyen
% 04/03/13

% [x,xmean,xstd] = standardize(x,1,[],[]);
% xt = standardize(xt,1,xmean,xstd);
y0 = y-mean(y);

likfunc = @likGauss;
covfunc = {@covSEard};
infFunc = @infExact;
% initial assignment of points to clusters (gp models)
[partition,~] = kmeans(x,K,'MaxIter',500,'Display','iter');
%[partition,~] = kmeans(x,K);
hyp = cell(K,1);
nmll = cell(K,1);
% initial cluster centroid
for k=1:K
  xk = x(partition==k,:);
  yk = y(partition==k); yk0 = yk-mean(yk);
  %yk0 = y0(partition==k);
  lengthscales = log((max(xk)-min(xk))'/2);
  lengthscales(lengthscales<-1e2)=-1e2;
  hyp{k}.cov = [lengthscales; 0.5*log(1e-4+var(yk0,1))];
  hyp{k}.lik = 0.5*log(1e-4+var(yk0,1)/4);
  hyp{k} = minimize(hyp{k},@gp,-100,infFunc,[],covfunc,likfunc,xk,yk0);
end

iter = 1; maxIter = 100;
predVar = zeros(size(x,1),K);
while iter <= maxIter
  disp(['iter ' num2str(iter)])
  % re-assign points to clusters
  for k=1:K
    [xk,yk0] = getPartition(x,y,partition,k);
    if size(xk,1) == 0
      predVar(:,k) = 1e10;
    else
      [~,~,~,predVar(:,k)] = gp(hyp{k},infFunc,[],covfunc,likfunc,xk,yk0,x);
    end  
  end
   % assign each point to model with smallest predVariance ('distance')
  [~,newpartition] = min(predVar,[],2);
  if sum(newpartition ~= partition) == 0
    disp('no new assignment of points to clusters. algorithm finishes');
    break;
  end
  disp([num2str(sum(newpartition ~= partition)) ' points re-allocated'])
  partition = newpartition;
  
  % update cluster centroids (actually only need to update cluster that
  % changes, but this is ok because the cluster is already at maxima)
  for k=1:K
    [xk,yk0]=getPartition(x,y,partition,k);
    [hyp{k},fk] = minimize(hyp{k},@gp,-100,infFunc,[],covfunc,likfunc,xk,yk0);
    nmll{k} = [nmll{k}; fk];
  end
  iter = iter+1;
end

Nt = size(xt,1);
testMean = zeros(Nt,K);
testVar = zeros(Nt,K);
for k=1:K
  [xk,yk0] = getPartition(x,y,partition,k);
  [~,~,testMean(:,k),testVar(:,k)] = gp(hyp{k},infFunc,[],covfunc,likfunc,xk,yk0,xt);
end
[~,testPartition] = min(testVar,[],2);
% global transformatino of output
% testMean = testMean + mean(y);
% fpred = zeros(Nt,1);
% for i=1:Nt
%   fpred(i) = testMean(i,testPartition(i));
% end
% smse = mysmse(yt,fpred,mean(y));

logpred = [];
smse = [];
for k=1:K
  [xk,yk0] = getPartition(x,y,partition,k);
  yk = y(partition==k);
  xtk = xt(testPartition==k,:);
  ytk = yt(testPartition==k);
  ytk0 = ytk-mean(yk);
  [~,~,fpredk,~,logpredk] = gp(hyp{k},infFunc,[],covfunc,likfunc,...
    xk,yk0,xtk,ytk0);
  logpred = [logpred; logpredk];
  fpredk = fpredk+mean(yk);
  smse = [smse; mysmse(ytk,fpredk,mean(yk))];
end
msll = -mean(logpred);
disp(smse)
smse = mean(smse);

function [xk,yk0] = getPartition(x,y,partition,k)
  %rng(1234,'twister');
  xk = x(partition==k,:);
  yk = y(partition==k); yk0 = yk-mean(yk);
  if size(xk,1) > 2000
    randInd = randperm(size(xk,1),2000);
    xk = xk(randInd,:); yk0 = yk0(randInd);
  end
