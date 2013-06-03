function label = assign_nearest_basis(W,x,nu)
%ASSIGN_NEAREST_BASIS label = assign_nearest_basis(W,x)
%  
% Assign x to partitions based on the basis inputs.
%   label(x_i) = argmin_k dist(x_i,partition_k) 
% where dist(x_i,partition_k) = min(dist(x_i,xu_k))
%
% Trung V. Nguyen
% 15/03/13
K = size(W,2);
dim = size(x,2);
% N x K = dist(x_i \in x, xu_k)
dist = zeros(size(x,1),K);
for k=1:K
  % the matrix inside min is dist(xu in xu_k, x_i in x), size nu x N
  dist(:,k) = min(sq_dist(reshape(W(:,k),nu,dim)',x'),[],1);
end
 % N x K distance from each point to partition centers
[~,label] = min(dist,[],2);

