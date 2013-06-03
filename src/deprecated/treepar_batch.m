clear; clc; close all;
% Batch running of dataset partitioning and fitc
basedir = '/home/trung/projects/datasets/kin40k';
basefile = 'kin40k';
[X,Y,Xtest,Ytest] = load_data(basedir, basefile);

%% Use regression tree to partition the dataset
min_leaf = 2000;
num_inducing = 500;
num_partitions = 4; % for kmeans

disp('fitting regression tree')
logger.minLeaf = min_leaf;
tree = RegressionTree.fit(X, Y, 'MinLeaf', min_leaf);
partitions = get_tree_partitions(tree);

fprintf('number of partitions: %d\n', size(partitions, 1));

disp('training fitc models')
models = train_gps_fitc(partitions,num_inducing);

%%
disp('assigning test points to partitions')
test_partitions = get_tree_partitions(tree, Xtest, Ytest);
for i=1:length(test_partitions)
  partitions{i}.xt = test_partitions{i}.x;
  partitions{i}.yt = test_partitions{i}.y;
end

disp('making predictions')
[smse msll] = predict_gps_fitc(models, partitions);
disp('smse, msll')
disp([mean(smse) mean(msll)])
