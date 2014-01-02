function ndcg = ndcg(y,fpred,k)
%NDCG ndcg = ndcg(y,fpred,k)
%   
% Computes the normalized discounted cumulative gain (NDCG):
%   ndcg(y) = dcg@k(y,fpred)/dcg@k(y,y)
% where
%   dcg@k(y,fpred) = \sum_i=1^k 2^y'_i - 1 / log(2+i)
% with y' being the permutation of y corresponding to sorted fpred.
%
% Trung Nguyen
% 10/12/13
if numel(y) < k
  k = numel(y);
end
[~,pi] = sort(fpred, 'descend');
dcg_f = sum((2.^y(pi(1:k)) - 1)./ log(2+(1:k)'));
[~,pi] = sort(y, 'descend');
dcg_y = sum((2.^y(pi(1:k)) - 1)./ log(2+(1:k))');
ndcg = dcg_f / dcg_y;
end

