function d = distance(X,c)
%DISTANCE d = distance(X,c)
%   
% Eucledian distance from the set of points in X to the centre c.
% 
%
C = repmat(c,size(X,1),1);
d = sum((X-C).^2,2);
end

