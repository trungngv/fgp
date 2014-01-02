function D = allDists(x,y,varargin)
% allDists: compute pairwise squared eucildean distances
% between each two column vectors in matrices x and y.
% If varargin is given and is an integer, then split y into 
% matrices of max size (varargin, size(y,1)) - to avoid 
% memory problems with large data amounts (we still assume the whole
% x-matrix fits in memory).
%
% For each vector in x, return the index of the vector in y closest to x.
%
% Author : Krzysztof Chalupka, University of Edinburgh 2011

%if size(varargin) == [1,1];
%    splitLen = varargin{:};
%else
splitLen = 1000; %size(x,1);
%end
D = zeros(1,size(y, 1));

for i=0:ceil(size(y,1)/splitLen)-1
  ys = y(i*splitLen+1:min((i+1)*splitLen,size(y,1)),:);
  Ds = full((-2)*(x*ys'));
  Ds = bsxfun(@plus,Ds,full(sum(ys.^2,2)'));
  Ds = bsxfun(@plus,Ds,full(sum(x.^2,2)));
  [phony, closest] = min(Ds);
  D(i*splitLen+1:min((i+1)*splitLen,size(y,1))) = closest;
end

