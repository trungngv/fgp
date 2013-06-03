% demo of inducing (basis) based partition
rng(1310,'twister');
func_assign = 'assign_nearest_center';
%func_assign = @assign_test;
datasetsDir = '/home/trung/projects/datasets/';
datasetName = 'pol';
fileName = 'pol';

K = 3;
num_inducing = 5;
numz = 20;
[x,y,xt,yt] = load_data([datasetsDir datasetName], fileName);
dim = size(x,2);

load motorcycle;
x = x_times;
y = accelaration;
dim = size(x,2);
xt = x;
yt = y;

[~,randind] = sort(rand(size(x,1),1));
randind = randind(1:(num_inducing*K));
randind = reshape(randind,num_inducing,K);
hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y,false));
w0 = [];
for k=1:K
  xu_k = x(randind(:,k),:);
  w0 = [w0; xu_k(:)];
  % TODO: different scale for each partition
  hyp = gpml_hyp_to_fitc(gpml_init_hyp(xu_k,y(randind(:,k)),false));
  w0 = [w0; hyp];
end

% learn basis points and hyp
tic
[w,fval] = minimize(w0,'prob_indpar_marginal',-500,x,y,K,num_inducing,numz);
toc

% prediction
W = reshape(w,numel(w)/K,K);
[fmean,logpred,~] = prob_indpar_predict(x,y,xt,yt,W,num_inducing);
fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y)));
fprintf('nlpd = %.4f\n', -mean(logpred));

