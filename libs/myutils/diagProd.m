function d = diagProd(A, B)
%D = DIAGPROD(A,B)
% Efficient computation of d = diag(A*B) where A is n x m and B is m x n.
d = sum(A.*B',2);
end
