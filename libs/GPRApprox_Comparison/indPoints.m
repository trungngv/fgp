function [out1, out2] = indPoints(x, N, method, covfunc, loghyper, varargin)

% indPoints - choose a subset of the data in x. Several methods
% are available: random choice, entropy maximization and farthest
% points clustering.
%
% usage: sod = indPoints(x, n, method, covfunc, loghyper)
%    or: [ci cc]      = indPoints(x, n, 'c', covfunc, loghyper)
%
% where:
%
% x        is an n by D matrix of datapoints
% N        is the number of points to be chosen
% method   is the method of choice. The choices available are:
%             'r'andom - choose randomly, return the indices of the SoD.
%             'e'ntropy - choose to minimize the predictive variance (the
%                         "Informative Vector Machine approach").
%             or 'c'lustering - perform Farthest Point Clustering and return
%                          the points in the dataset x closest to the FPC cluster
%                          centers (if one output out1 is requested) or return
%                          the cluster indices in out1 and cluster centers in out2.
% covfunc  is the covariange function used in entropy maximization
%             (needs not be provided with other methods)
% logtheta is a column vector of hyperparameters to the covariance 
%             functions (also only with entropy maximization)
% 
% For help on covariance functions see "help covFunctions".

[n, D] = size(x);

if method == 'r'
  perm = randperm(n);
  out1 = perm(1:N);

elseif method == 'e'
  % Select the set of N points that maximize the
  % differential entropy of the distribution.

  id  = ceil(n.*rand());
  sod = [id];

  % Assume the self-covariances are equal
  % (this only works for stationary kernels, but 
  % should be easy to change).
  Kss = feval(covfunc{:}, loghyper.cov, x, 'diag');
  %Kss     = selfcov * ones(n, 1)

  % Iteratively select elements with greatest variances.
  for i = 1:(N-1)

    if i ~= 1
      vi             = v(:, id);
      li             = sqrt(Kss(1) - p(id));
      covs   = feval(covfunc{:}, loghyper.cov, x, x(id,:));
      covs(id) = Kss(1);
      v = [v; (covs - v' * vi)'/li];
      p = p + v(i,:)'.^2;
    else
      % These equations (here run on the first iterations only) 
      % are solved on every iteration, using dynamic 
      % programming to avoid the cubic time cost.
      L              = chol(Kss(1));
      covs   = feval(covfunc{:}, loghyper.cov, x(id,:), x);
      covs(id)       = Kss(1);
      v              = L\covs;
      p              = sum(v.*v,1)';
    end

    vars = Kss - p;
    [var, id] = max(vars);
    sod = [sod id];
  end
  out1 = sod;

elseif method == 'c'
  % Select a set of N datapoints
  % as the centers of N clusters using 
  % a k-center clustering algorithm.
  [rx, ci, cc, np, cr] = KCenterClustering(D, n, x', N);

  % Find the indices of points closest to each center.
  if length(varargin) > 0
    splitLen = varargin{:};
  else
    splitLen = size(x,1);
  end
  if nargout == 1
    out1 = allDists(x, cc', splitLen);
  else
    % Return the indices of the clusters
    % to which points in x belong and the 
    % cluster centers (note those need not 
    % be in x!)
    out1 = ci;
    out2 = cc;
  end
end
