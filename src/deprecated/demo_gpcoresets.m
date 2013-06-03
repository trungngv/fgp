datasetsDir = '/home/trung/projects/datasets/';
% datasetName = 'mysynth';
% fileName = 'synth1';
datasetNames = {'kin40k'};%,'pumadyn32nm','myelevators','pol'};
fileName = datasetNames;
date = date(); timestamp = tic;
for i=1:length(datasetNames)
  [x,y,xt,yt] = load_data([datasetsDir datasetNames{i}], fileName{i});
  x = x(1:2000,:); y = y(1:2000,:);
  [smse,msll,partition,hyp,nmll,maxVar] = gpcoresets(5,x,y,xt,yt);
  save(['/home/trung/projects/ensemblegp/output/gpcoresets-' datasetNames{i}...
    '-' date '-' num2str(timestamp) '.mat'],'smse','msll','partition','hyp','nmll');
end

