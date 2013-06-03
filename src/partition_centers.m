function centers = partition_centers(Xu,num_inducing,K,dim)
%PARTITION_CENTERS centers = partition_centers(Xu,num_inducing,K,dim)
%  Compute the partition centers given the inducing points.
%
% INPUT
%   - Xu : (num_inducing x dim) x K, parametrisation of the inducing poitns
%   - num_inducing: number of inducing points per partition
%   - K : number of partitions
%   - dim : input dimension
% OUTPUT
%   - centers : K x dim centers
%
% 08/04/13
% Trung Nguyen
centers = zeros(K,dim);
for k=1:K
  centers(k,:) = mean(reshape(Xu(:,k),num_inducing,dim));
end

