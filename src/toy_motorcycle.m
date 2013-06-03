% demo of inducing (basis) based partition
%rng(131,'twister');
func_assign = 'zmap_mahala';

K = 2;
num_inducing = 20;
optim_xu = cell(K,1);
load motorcycle;
dim = size(x,2);
xt = (min(x):1:max(x))';
yt = zeros(size(xt,1),1);

figure;
plot(x,y,'xr', 'MarkerSize', 8);
hold on;

%kmeans and rpc works well
randind = init_rpc(x,K,num_inducing);
hyp = gpml_hyp_to_fitc(gpml_init_hyp(x,y,false));
w0 = [];
colors = {'g','r','k'};
for k=1:K
  xu_k = x(randind(:,k),:);
  plot(xu_k, -150*ones(size(xu_k)),[colors{k} 'x'],'MarkerSize',18)
  w0 = [w0; xu_k(:)];
  %hyp = gpml_hyp_to_fitc(gpml_init_hyp(xu_k,y(randind(:,k)),false));
  w0 = [w0; hyp];
end
% learn basis points and hyp
tstart = tic;
%label = [];
%[w,fval] = minimize(w0,'indpar_marginal',1000,x,y,K,num_inducing,func_assign);
[w,fval,label] = msgp_train(w0,x,y,K,num_inducing,50,10);
toc(tstart)

% assign points to partition based on the basis points
W = reshape(w,numel(w)/K,K);
if isempty(label)
  label = feval(func_assign,W(1:end-dim-2,:),x,num_inducing);
end  
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
      x(indk,:),y(indk),xt(test_indk,:),yt(test_indk),xu,hypk,false);
  catch eee
    valid(k)= false;
  end
end

% % standard gp:
% sgp = standardGP([],x,y,xt,[],0);
% samples = [];
% for i=1:numel(xt)
%   yi = fmean(i) + sqrt(fvar(i)) .* randn(100,1);
%   samples = [samples; [repmat(xt(i),100,1) + 0.2 - 0.4 * rand(100,1),yi]];
% end
% plot(xt,sgp.fmean + 2*sqrt(sgp.fvar),'-r','LineWidth',1);
% plot(xt,sgp.fmean - 2*sqrt(sgp.fvar),'-r','LineWidth',1);

plot(xt,fmean,'-k','LineWidth',2);
for i=1:numel(xt)
  yi = fmean(i) + sqrt(fvar(i)) .* randn(100,1);
  plot(repmat(xt(i),100,1) + 0.2 - 0.4 * rand(100,1),yi,'.b','MarkerSize', 2);
end
axis([min(x),max(x),-150,100]);
xlabel('Time (ms)')
ylabel('Acceleration (g)')
box off;
set(gca, 'FontSize', 20);
% saveas(gcf,'motorcycle.png','png')
% saveas(gcf,'motorcycle.eps','epsc')

fprintf('fval = %.4f\n', fval(end));
fprintf('smse = %.4f\n', mysmse(yt,fmean,mean(y))); % must use non-transformed y
fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(fmean-yt)));
fprintf('sq diff = %.4f\n', sqrt(mean((fmean-yt).^2))); 
fprintf('nlpd = %.4f\n', -mean(logpred));

%% plotting for motorcycle
allc = zeros(K,1);
for k=1:K
  plot(optim_xu{k}, 100*ones(size(optim_xu{1})),[colors{k} 'x'],'MarkerSize',18)
  allc(k) = mean(optim_xu{k});
end

sortedc = sort(allc);
for k=1:K-1
  plot(0.5*[sortedc(k)+sortedc(k+1), sortedc(k)+sortedc(k+1)],[-150,100],'-');
end

figure;
plot(1:numel(fval),fval);

%% plotting for motorcycle
% load motorcycle;
% figure; hold on;
% plot(x,y,'+','MarkerSize',12);
% set(gca,'FontSize',30)
% xlabel('x')
% ylabel('f(x)');
% hold on;
% xx=[15,15];
% yy=[-150,100];
% plot(xx,yy,'r-.','LineWidth',3)
% x3 = 2:1.7:12;
% y3 = -150*ones(size(x3));
% plot(x3,y3,'rx','MarkerSize',12)
% x4 = [22:24,33:35,42:44];
% y4 = -150*ones(size(x4));
% plot(x4,y4,'rx','MarkerSize',12)
