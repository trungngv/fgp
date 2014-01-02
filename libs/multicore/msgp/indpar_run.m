% demo of inducing (basis) based partition
rng(1310,'twister');
func_assign = 'zhat_mahala';
%func_assign = @assign_test;
datasetsDir = '/home/trung/projects/datasets/';
datasetName = 'song100k';
fileName = 'song100k';

global globalx;
global globaly;

K = 3;
num_inducing = 7;
optim_xu = cell(K,1);
% [globalx,globaly,xt,yt] = load_data([datasetsDir datasetName], fileName);
dim = size(globalx,2);

% housekeeping for slave to read later (must do for each batch)
% todo: make function
% x = globalx; y = globaly;
% globaly = globaly-mean(globaly); % zero-mean for training
% save('tnvglobals.mat','x','y');
% disp('saved file')

load motorcycle;
dim = size(x,2);
xt = x;
yt = y;
xt = linspace(min(x),max(x),300)';
yt = zeros(size(xt,1),1);

figure;
plot(x,y,'x')
hold on;
globalx = x; globaly = y;

randind = init_fpc(globalx,K,num_inducing);
hyp = gpml_hyp_to_fitc(gpml_init_hyp(globalx,globaly,false));
w0 = [];
colors = {'g','r','k'};
for k=1:K
  xu_k = globalx(randind(:,k),:);
  plot(xu_k, -150*ones(size(xu_k)),[colors{k} 'x'],'MarkerSize',18)
  w0 = [w0; xu_k(:)];
  hyp = gpml_hyp_to_fitc(gpml_init_hyp(xu_k,globaly(randind(:,k)),false));
  w0 = [w0; hyp];
end
% learn basis points and hyp
tstart = tic;
[w,fval] = minimize(w0,'indpar_marginal',500,globalx,globaly,K,num_inducing,func_assign);
%[w,fval] = minimize(w0,'indpar_marginal_fast',100,K,num_inducing,func_assign);
toc(tstart)

% assign points to partition based on the basis points
W = reshape(w,numel(w)/K,K);
label = feval(func_assign,W(1:end-dim-2,:),globalx,num_inducing);
test_label = feval(func_assign,W(1:end-dim-2,:),xt,num_inducing);

% predict
Nt = size(xt,1);
valid = true(K,1);
fmean = zeros(Nt,1); fvar = zeros(Nt,1); logpred = zeros(Nt,1);
for k=1:K
  indk = label == k;
  test_indk = test_label == k;
  wk = W(:,k);
  %TODO: use a unboxing function
  xu = reshape(wk(1:end-dim-2),num_inducing,dim);
  optim_xu{k} = xu;
  hypk = fitc_hyp_to_gpml(wk,dim);
  try
    [fmean(test_indk),fvar(test_indk),logpred(test_indk)] = gpmlFITC(...
      globalx(indk,:),y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,false);
  catch eee
    valid(k)= false;
  end
end
plot(xt,fmean,'-');
t1=fmean(:);
t2=yt(:);

fprintf('fval = %.4f\n', fval(end));
fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y))); % must use non-transformed y
fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2))); 
fprintf('nlpd = %.4f\n', -mean(logpred));

%% plotting for motorcycle
allc = zeros(K,1);
for k=1:K
  plot(optim_xu{k}, 120*ones(size(optim_xu{1})),[colors{k} 'x'],'MarkerSize',18)
  allc(k) = mean(optim_xu{k});
end

sortedc = sort(allc);
for k=1:K-1
  plot(0.5*[sortedc(k)+sortedc(k+1), sortedc(k)+sortedc(k+1)],[-150,100],'-')
end

