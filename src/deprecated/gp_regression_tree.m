clear; clc; close all;

%% synthetic dataset
% N = 1000;
% Ntest = 100;
% X = rand(N, 3);
% truef = @(X) X(:,1).^2 + sin(X(:,2)) + 0.5*X(:,3);
% Y = truef(X) + 0.1*randn(N,1);
% Xtest = rand(Ntest, 3);
% Ytest = truef(Xtest);

%% datasets in sparse spectrum
dataset = 'elevators';
logger.dataset = dataset;
load(dataset)
X = X_tr;
Y = T_tr;
Xtest = X_tst;
Ytest = T_tst;

logger.X = X;
logger.Y = Y;
logger.Xtest = Xtest;
logger.Ytest = Ytest;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Init GP model for hyper-parameters
disp('training a init model')
iinit = randperm(length(X), 1024);
Xinit = X(iinit,:);
Yinit = Y(iinit,:);
initModel = standardGP([], Xinit, Yinit);

%% REGRESSION TREE TRAINING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('fitting regression tree')
minLeaf = 500;
logger.minLeaf = minLeaf;
tree = RegressionTree.fit(X, Y, 'MinLeaf', minLeaf);

%tree = prune(tree, 'level', 4);
%view(tree, 'mode', 'graph');

%% training gps for tree partitions
disp('training models for tree partitions')
partitions = get_tree_partitions(tree);
fprintf('number of partitions: %d\n', size(partitions, 1));
% normalise data in each partition
for i=1:length(partitions)
  [partitions{i}.X, partitions{i}.xmean, partitions{i}.xstd] = standardize(partitions{i}.X,1,[],[]);
  [partitions{i}.Y, partitions{i}.ymean, partitions{i}.ystd] = standardize(partitions{i}.Y,1,[],[]);
end
models = train_gps(initModel, partitions);

%% prediction with weighted combination of multiple gps
disp('making weighted prediction')
[ymu, yvar] = gp_predict_weighted(models, partitions, Xtest);
logger.tree.weightedgp.smse = mysmse(Ytest, ymu);
logger.tree.weightedgp.mae = mean(abs(Ytest - ymu));
fprintf('weighted combination smse = %.4f\n', logger.tree.weightedgp.smse);
fprintf('weighted combination mae = %.4f\n', logger.tree.weightedgp.mae);
%plot(Xtest, ymu, '-b');

%% prediction with hard assignment to cluster
disp('making hard assignment prediction')
testParts = get_tree_partitions(tree, Xtest, Ytest);
for i=1:length(testParts)
  partitions{i}.Xtest = testParts{i}.X;
  partitions{i}.Ytest = testParts{i}.Y;
end
[yMeans, yVars] = gp_predict_hard_assignment(models, partitions);
ytest = []; ymu = [];
for i=1:length(partitions)
 ytest = [ytest; partitions{i}.Ytest];
 ymu = [ymu; yMeans{i}];
end
logger.tree.hardgp.smse = mysmse(ytest, ymu);
logger.tree.hardgp.mae = mean(abs(ytest - ymu));
fprintf('hard assignment smse = %.4f\n', logger.tree.hardgp.smse);
fprintf('hard assignment mae = %.4f\n', logger.tree.hardgp.mae);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prediction with tree
disp('making prediction with regression tree')
ymean = predict(tree, Xtest);
logger.rt.smse = mysmse(Ytest, ymean);
logger.rt.mae = mean(abs(Ytest - ymean));
fprintf('treed smse = %.4f\n', logger.rt.smse);
fprintf('treed mae = %.4f\n', logger.rt.mae);

save(['ensemblegp/output/regressionTreeGPs-', logger.dataset, '-minLeaf', num2str(logger.minLeaf)], 'logger');
