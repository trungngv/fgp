function Z = sample_z(numz,Xu,x,num_inducing,centers)
%SAMPLE_Z Z = sample_z(numz,Xu,x,num_inducing,centers)
%   Generate samples from p(z|Xu)
%
% INPUT
%   - num_samples : number of samples to generate
%   - Xu : (num_inducing x dim) x K inducing inputs
%   - x : (N x dim) training inputs
%   - nz : number of samples
%   - centers : optional
%
% OUTPUT
%   - Z : (nz x N) the samples where each row is a sample
%
% Trung Nguyen
% 08/04/13

%TODO: try multiple distance functions
rng(2011,'twister');
K = size(Xu,2);
[N dim] = size(x);
if nargin == 4
  centers = partition_centers(Xu,num_inducing,K,dim);
end  
% N x K distance from each point to partition centers
Z = zeros(numz,N);

% maximize the difference while maintaining numerical safety
tmp = sq_dist(x',centers');
beta = log(200)/log(max(max(tmp))); % ensure numerically safe
dist = exp(-tmp.^beta);   % first way
%beta = 1;
% dist = exp(beta./sq_dist(x',centers'));  % second way
% dist(dist == Inf) = 1;
% dist = beta./sq_dist(x',centers');     % third way

% discrete probabilities (one in a row for each training point)
prob = dist ./ repmat(sum(dist,2),1,K);
%disp(prob(1:5,:))
% cumulative prob to sample from discrete probabilities
cumprob = [zeros(N,1), cumsum(prob,2)];
for iz=1:numz
  nsamples = rand(N,1);
  Z(iz,:) = rowfind(cumprob>repmat(nsamples,1,K+1))-1;
end
