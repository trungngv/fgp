%rng(131,'twister');
load motorcycle;

% test locations
xt = (min(x):1:max(x))';    
yt = zeros(size(xt,1),1);

% Model configuration
Z_ASSIGN = 'zhat_mahala';   % assignment function
K = 2;  % number of experts
NUM_INDUCING = 20;  % number of inducing points per expert
INIT_INDUCING = 'init_random'; % one of init_fpc, init_kmeans, init_rpc, and init_random
randind = feval(INIT_INDUCING,x,K,NUM_INDUCING);
hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y,false));
w0 = [];
for k=1:K
  xu_k = x(randind(:,k),:);
  w0 = [w0; xu_k(:)];
  w0 = [w0; hyp];
end

% TRAINING (learning inducing inputs and hyperparameters)
tstart = tic;
[w,fval,label] = msgp_train(w0,x,y,K,NUM_INDUCING,50,10);
toc(tstart)

% assign test points to partition
dim = size(x,2);
W = reshape(w,numel(w)/K,K);
test_label = feval(Z_ASSIGN,W(1:end-dim-2,:),xt,NUM_INDUCING);

% PREDICTION
[smse,nlpd,mae,sqdiff,fmean,fvar,logpred,~,~] = msgp_predict(...
  NUM_INDUCING,x,xt,y,yt,0,W,label,test_label,true);

