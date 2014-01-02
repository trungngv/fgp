%% results latest

datasets = {'kin40k','pumadyn32nm','myelevators','pol'};
% Nu, K, SMSE, NLPD, obj, training => (Nu,SMSE, NLPD,Training)
kin40k = [1500	1	0.0566	-0.9925	-7839.77	9997.5
          750	2	0.0517	-0.7515	-5323.41	2444.4
          500	3	0.0539	-0.6385	-4298.38	1664.9
          750	2	0.0757	-0.3591	-6384.94	5224.8
          500	3	0.0558	-0.7085	-5196.31	4212.6
          750	2	0.0625	-0.5518	-3998.47	5123.8
          500	3	0.0772	-0.2903	-1829.1	3384.4];
kin40k = kin40k(:,[1,3,4,6]);
kin40k(:,4) = kin40k(:,4) / 60; % to minutes

puma = [1500	1	0.0441	-0.163	-1005.42	15170.2
750	2	0.0501	-0.0575	-1139.05	2944.4
500	3	0.0531	-0.006	-916.17	2322.8
750	2	0.0473	-0.1022	-1369.15	7702.8
500	3	0.0494	-0.0724	-1730.27	5708.5
750	2	0.0524	0.0457	-1366.91	7627.6
500	3	0.0528	0.0469	-1814.42	5093.2];
puma = puma(:,[1,3,4,6]);
puma(:,4) = puma(:,4) / 60;

elevators = [1500	1	0.1323	-4.7199	-41788.53	10986.2
750	2	0.143	-4.4854	-38799.62	687.3
500	3	0.1468	-4.4765	-38666.24	1139.5
750	2	0.151	-4.6446	-40740.84	6241.7
500	3	0.56	-4.284	-39768.29	4028.7
750	2	0.1477	-4.646	-40648.26	6136.4
500	3	0.1527	-4.6153	-39764.77	3750.1];
elevators = elevators(:,[1,3,4,6]);
elevators(:,4) = elevators(:,4) / 60;

pol = [1500	1	0.0259	3.2974	20077.06	17612.4
750	2	0.0284	1.9914	20646.28	8783.8
500	3	0.027	2.2584	22661.82	6254.7
750	2	0.037	3.711	13724.81	8635.3
500	3	0.4636	7.6828	13091.59	7690.2
750	2	0.0384	2.2118	23398.59	8644.9
500	3	0.0282	2.5129	25719.7	6007.2];
pol = pol(:,[1,3,4,6]);
pol(:,4) = pol(:,4) / 60;

figure;
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0.1 0.1 20.5 10.5]);
set(gcf, 'PaperSize', [21 11]);

inducings = [1500,750,500];
sizes = [50,50,50];
specifiers = {'s','c','^'};
linestyles = {'-','-','-'};
colors = {'r','g','b'};

%(Nu,SMSE, NLPD,Training)
data = {kin40k,puma,elevators,pol};
trainingIdx  = 4;
yidx = 2; % smse
%yidx = 3; % nlpd 
%plot for smse
for idata=1:4
  pdata = data{idata};
  subplot(1,4,idata);
  hold on;
  for i=1:3
    ind = pdata(:,1) == inducings(i);
    scatter(pdata(ind,trainingIdx),pdata(ind,yidx),sizes(i),'k',specifiers{i},'filled');
  end
  %our method
  line(pdata(1:3,trainingIdx),pdata(1:3,yidx),'LineWidth',2,'Color',colors{1});
  %kmeans
  line(pdata([1,4:5],trainingIdx),pdata([1,4:5],yidx),'LineWidth',2,'Color',colors{2});
  %random
  line(pdata([1,6:7],trainingIdx),pdata([1,6:7],yidx),'LineWidth',2,'Color',colors{3});
  xlabel('Training time (mins)')
  ylabel('Standardized Mean Square Error (SMSE)')
end
box off;
legend('boxoff');
legend({'M=1500', 'M=750', 'M=500', 'our method', 'kmeans', 'random'},...
  'Location','NorthEast');

%%
% dist = [0.25 0.4 0.25 0.1];
% nz = 5000;
% [N K] = size(dist);
% Z = zeros(nz,N);
% prob = dist ./ repmat(sum(dist,2),1,K);
% cumprob = [zeros(N,1), cumsum(prob,2)];
% for inz=1:nz
%   nsamples = rand(N,1);
%   Z(inz,:) = rowfind(cumprob>repmat(nsamples,1,K+1))-1;
% end
% figure;
% hist(Z)
% 
