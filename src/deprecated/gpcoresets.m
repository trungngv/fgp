function [smse,msll,partition,hyp,nmll,maxVar] = gpcoresets(K,x,y,xt,yt)
%GPKMEANS gpcoresets(x,y,xt,yt)
%   
% GP via k-coresets 
% 
% Trung V. Nguyen
% 06/03/13
likfunc = @likGauss; covfunc = {@covSEard}; infFunc = @infExact;
y0 = y-mean(y);

% initial clusters (include 30 points each)
Minit = 30;
[partition,centres] = kmeans(x,K,'MaxIter',500,'Display','iter');
hyp = cell(K,1);
nmll = cell(K,1);
% initial cluster centroid
for k=1:K
  % find Minit points closest to the k-means centroid
  indk = find(partition==k);
  partition(indk) = 0;                            % cancel cluster assignment
  dist2c = distance(x(indk,:),centres(k,:));
  [~,sortedInd] = sort(dist2c);
  indk = indk(sortedInd(1:Minit));
  partition(indk) = k;                           % cluster assignment of core
  xk = x(indk,:);   yk = y(indk); yk0 = yk-mean(k);
  % find the 'centre' of this cluster (i.e. optimum gp for this cluster)
  lengthscales = log((max(xk)-min(xk))'/2);
  lengthscales(lengthscales<-1e2)=-1e2;
  hyp{k}.cov = [lengthscales; 0.5*log(1e-4+var(yk0,1))];
  hyp{k}.lik = 0.5*log(1e-4+var(yk0,1)/4);
  hyp{k} = minimize(hyp{k},@gp,-100,infFunc,[],covfunc,likfunc,xk,yk0);
end

iter = 1; maxIter = size(x,1)-Minit*K; Nsamples = 70; maxVar = zeros(maxIter,1);
while iter <= maxIter
  disp(['iter ' num2str(iter)])
  % select point further away from centres (point of highest variance)
  outsideInd = find(partition == 0);
  if size(outsideInd,1) <= Nsamples,  sampleInd = outsideInd;
  else   sampleInd = randsample(outsideInd,Nsamples); end
  predVar = zeros(size(sampleInd,1),K);
  % first compute distance to centres (pred variance by clusters)
  for k=1:K
    [xk,yk0] = getPartition(x,y,partition,k);
    [~,~,~,predVar(:,k)] = gp(hyp{k},infFunc,[],covfunc,likfunc,xk,yk0,x(sampleInd,:));
  end
  % then find the furthest point and assign to a partition
  [predVar,label] = min(predVar,[],2);
  [maxVar(iter),maxidx] = max(predVar);
  furthestInd = sampleInd(maxidx);
  k = label(maxidx);
  partition(furthestInd) = k;
  % update the centre of the partition of include new point
  [xk,yk0]=getPartition(x,y,partition,k);
  [hyp{k},fk] = minimize(hyp{k},@gp,-5,infFunc,[],covfunc,likfunc,xk,yk0);
  nmll{k} = [nmll{k}; fk];
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
logpred = [];
smse = [];
for k=1:K
  [xk,yk0] = getPartition(x,y,partition,k);
  yk = y(partition==k);
  xtk = xt(testPartition==k,:);
  ytk = yt(testPartition==k);   ytk0 = ytk-mean(yk);
  [~,~,fpredk,~,logpredk] = gp(hyp{k},infFunc,[],covfunc,likfunc,...
    xk,yk0,xtk,ytk0);
  logpred = [logpred; logpredk];
  fpredk = fpredk+mean(yk);
  smse = [smse; mysmse(ytk,fpredk,mean(yk))];
end
msll = -mean(logpred);
disp(smse)

function [xk,yk0] = getPartition(x,y,partition,k)
  %rng(1234,'twister');
  xk = x(partition==k,:);
  yk = y(partition==k); yk0 = yk-mean(yk);
  if size(xk,1) > 2000
    randInd = randperm(size(xk,1),2000);
    xk = xk(randInd,:); yk0 = yk0(randInd);
  end
