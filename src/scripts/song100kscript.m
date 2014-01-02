clear;
N = 20000;
split_idx = 463715;
data = csvread('YearPredictionMSD.csv');
ind = 1:N; % still respect the distribution
x = data(ind,2:end);
y = data(ind,1);
xt = data(split_idx+1:end,2:end);
yt = data(split_idx+1:end,1);
% for small test set (same dist)
xt = xt(1:10000,:); yt = yt(1:10000);
save_data(x,y,xt,yt,'datasets/song20k','song20k');