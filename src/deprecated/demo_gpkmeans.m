datasetsDir = '/home/trung/projects/datasets/';
%  datasetName = 'mysynth';
%  fileName = 'synth1';
datasetName = 'pumadyn32nm';
fileName = 'pumadyn32nm';
date = date(); timestamp = tic;
[x,y,xt,yt] = load_data([datasetsDir datasetName], fileName);
x = x(1:2000,:); y = y(1:2000,:);
[smse,msll,partition,hyp,nmll] = gpkmeans(5,x,y,xt,yt);
save(['/home/trung/projects/ensemblegp/output/gpkmeans-' date '-' num2str(timestamp) '.mat'],...
  'smse','msll','partition','hyp','nmll');


