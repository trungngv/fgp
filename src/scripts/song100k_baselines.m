datasets_dir = '/home/trung/projects/datasets/';
dataset = 'song100k';
[x,y,xt,yt] = load_data([datasets_dir dataset], dataset);
disp('finished reading data')

%% linear regression
y0 = y-mean(y);
tic;
model = LinearModel.fit(x,y0);
disp(['linear regression time ', num2str(toc)]);
ypred = predict(model,xt)+mean(y);
fprintf('smse = %.4f\n', mysmse(yt,ypred,mean(y)));
fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(ypred-yt)));
fprintf('sq diff = %.4f\n', sqrt(mean((ypred-yt).^2)));

%figure; hist(abs(ypred-yt),200);

%% constant prediction (mean)
%ypred=repmat(mean(y),size(yt,1),1);
%fprintf('smse = %.4f\n', mysmse(yt,ypred,mean(y)));
%fprintf('avg absolute diff (mae) = %.4f\n', mean(abs(ypred-yt)));
%fprintf('sq diff = %.4f\n', sqrt(mean((ypred-yt).^2)));

%% nearest neighbors
%k1errors = zeros(5,3);
%k50errors = zeros(5,3);
%seed = 1110;
%for i=1:5
%rng(seed+i,'twister');
%K = 1;
disp(['knn k = ' num2str(K)]);
%ypred = knnclassify(xt,x,y,K);
k1errors(i,:) = [mysmse(yt,ypred,mean(y)),mean(abs(ypred-yt)),sqrt(mean((ypred-yt).^2))];
fprintf('smse = %.4f\n', k1errors(i,1));
fprintf('mae = %.4f\n', k1errors(i,2));
fprintf('sq diff = %.4f\n', k1errors(i,3));

%smse = 1.6833
%avg absolute diff (mae) = 9.9080
%sq diff = 14.0897

K = 50;
disp(['knn k = ' num2str(K)]);
%ypred = knnclassify(xt,x,y,K);
k50errors(i,:) = [mysmse(yt,ypred,mean(y)),mean(abs(ypred-yt)),sqrt(mean((ypred-yt).^2))];
fprintf('smse = %.4f\n', k50errors(i,1));
fprintf('mae = %.4f\n', k50errors(i,2));
fprintf('sq diff = %.4f\n', k50errors(i,3));
%end
% smse = 1.3326
% mae = 8.2084
% sqdiff = 12.5365
%save('song100k_knn.mat', 'k1errors', 'k50errors');

