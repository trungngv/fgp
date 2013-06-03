function gradchek_prob_inpar_marginal()
%GRADCHEK_PROB_INPAR_MARGINAL gradchek_prob_inpar_marginal()
%  Gradient check for the function prob_inpar_marginal

rng(1110,'twister');
N = 5; dim = 1;
x = rand(N,dim); y = rand(N,1);
K = 2; num_inducing = 2; numz = 50;
w = [];
for k=1:K
  xu_k = rand(num_inducing, dim);
  w = [w; xu_k(:)];
  w = [w; rand(dim+2,1)];
end

[gradient, delta] = gradchek(w', @f, @grad,x,y,K,num_inducing,numz);
disp('difference')
fprintf('%12.12f\n', delta);

end

function val = f(w,x,y,K,num_inducing,numz)
  val = prob_indpar_marginal(w',x,y,K,num_inducing,numz);
end

function grad = grad(w,x,y,K,num_inducing,numz)
  [~, grad] = prob_indpar_marginal(w',x,y,K,num_inducing,numz);
  grad = grad';
end


