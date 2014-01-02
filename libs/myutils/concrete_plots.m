[x,y,xtest,ytest] = load_data('projects/gprn/data/concreteslump', 'concrete3');
all_x = [x; xtest];
all_y = [y; ytest];
figure;
scatter(all_y(:,1),all_y(:,2));
figure;
scatter(all_y(:,2),all_y(:,3));
figure;
scatter(all_y(:,3),all_y(:,1));
