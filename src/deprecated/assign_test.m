function label = assign_test(W,x,nu)
%ASSIGN_TEST Summary of this function goes here
%   Detailed explanation goes here
rng(1110,'twister');
label = randsample(size(W,2),size(x,1),true);

end

