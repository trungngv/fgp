%% results latest
%datasets = {'kin40k','pumadyn32nm','pol'};
datasets = {'song100k'};
out = '/home/trung/projects/fagpe/output/song100k_gpsvi.csv';
f = fopen(out,'w');
%fprintf(f,'dataset,Nu,K,smse,nlpd,objective,training(s),seed,valid\n');
% song100k
fprintf(f,'smse,mae,sqdiff,nlpd,training(s),seed\n');
timestamps = fieldnames(logger);
for j=1:numel(datasets)
%  for Nu=[750]
%    for k=[2]
      for i=1:numel(timestamps)
       % fname = datasets{j}
%         fname = [datasets{j} 'NU' num2str(Nu) 'k' num2str(k)];
%        fname = [datasets{j} 'M' num2str(Nu) 'k' num2str(k)];
%        if isfield(logger.(timestamps{i}),fname)
%          s = logger.(timestamps{i}).(fname);
%           %---------- for kmeans and rand
%           s.obj = 0;
%           for ik=1:k
%             s.obj = s.optim_models{ik}.obj + s.obj;
%           end
%           %----------
        s = logger.(timestamps{i});
%        fprintf(f,'%.4f,%.4f,%.2f,%.1f,%d,%d\n',...
 %         s.smse,s.nlpd,s.obj(end),s.training_time,logger.(timestamps{i}).rng,s.valid);
%        figure;
%        plot(1:200,s.obj(1:200));
    %    title(datasets{j});
%           fprintf(f,'%s,%d,%d,%.4f,%.4f,%.2f,%.1f,%d,%d\n',datasets{j},Nu,k,...
%             s.smse,s.nlpd,0,s.training_time,logger.(timestamps{i}).rng,s.valid);
        % for song100k
         fprintf(f,'%.4f,%.4f,%.4f,%.4f,%.1f,%d\n',...
           s.smse,s.mae,s.sqdiff,s.nlpd,s.training_time,logger.(timestamps{i}).rng);
%        end
        end
%    end
%  end
end

fclose(f);

%% results for partitinoing
% NOTE: if smse = 0 or msll = 0, more partitions than necessary.
%load('/home/trung/projects/ensemblegp/output/partition-k2-nu500-26-Mar-2013-1364287690313529.mat')
%disp('kin40k')
%disp([logger.kin40k.smse, logger.kin40k.msll])
%disp('puma')
%disp([logger.pumadyn32nm.smse, logger.pumadyn32nm.msll])
%disp('pol')
%disp([logger.pol.smse, logger.pol.msll])
%disp('elevators')
%disp([logger.myelevators.smse, logger.myelevators.msll])

%disp('mean')
%disp([mean(logger.kin40k.smse), mean(logger.kin40k.msll)])
%disp([mean(logger.pumadyn32nm.smse), mean(logger.pumadyn32nm.msll)])
%disp([mean(logger.pol.smse), mean(logger.pol.msll)])
%disp([mean(logger.myelevators.smse), mean(logger.myelevators.msll)])
% 
%disp('objective')
%disp(logger.kin40k.obj)
%disp(logger.pumadyn32nm.obj)
%disp(logger.pol.obj)
%disp(logger.myelevators.obj)

