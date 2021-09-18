%% prep data
clear grp; grp = ['all'];   %HCs, OCD, or all
if grp == 'HCs'
    subjlist = subjList1;
elseif grp == 'OCD'
    subjlist = subjList2;
else
    subjlist = [subjList1 subjList2];
end
maskROIs = {'striatum', 'pallidum', 'thalamus'};
maskWBs = {'WBstr', 'WBpal', 'WBthal'};

% subject data
for i = 1:length(subjlist)
    img_data{i} = load_nii([dataDir '/preproc_conn_2mm_TMT/niftiDATA_Subject' subjlist{i} '_Condition000.nii']); % 2mm vox dim
    img_data{i} = img_data{i}.img;
end

% roi mask
for g = 1:length(maskROIs)
    mask{g} = load_nii([dataDir '/HO_atlas_thr50/HO_' maskROIs{g} '_2mm_bin.nii']);
    maskCoord{g} = mask{g}.img(:,:,:) ~= 0;     % extract non-zero values
    ind{g} = find(maskCoord{g});
    sz{g} = size(maskCoord{g});
    [x{g},y{g},z{g}] = ind2sub(sz{g},ind{g});   % assign mask coordinates of each ROIs
    % caution to interpret the matrix coordinates in FSL MNI space;
    % LR inverted, FSL coordinate starts from zero
end

% whole-brain mask
for gg = 1:length(maskWBs)
    mask_wb{gg} = load_nii([dataDir '/HO_atlas_thr50/HO_', maskWBs{gg} '_2mm_bin.nii']);
    ind_wb{gg} = find(mask_wb{gg}.img);
    sz_wb{gg} = size(mask_wb{gg}.img);
    [x_wb{gg},y_wb{gg},z_wb{gg}] = ind2sub(sz_wb{gg},ind_wb{gg});
end
clear mask maskCoord mask_wb


%% correlation between ROI voxels and target whole-brain voxles
clear roi voxroi voxWB
roi = 2;    % striatum - 1; pallidum - 2; thalamus - 3
for i = 1:length(subjlist)
    for k = 1:length(ind{roi})
        voxroi{i}(1:112,k) = img_data{i}(x{roi}(k),y{roi}(k),z{roi}(k),1:112);
    end
    for kk = 1:length(ind_wb{roi})
        voxWB{i}(1:112,kk) = img_data{i}(x_wb{roi}(kk),y_wb{roi}(kk),z_wb{roi}(kk),1:112);
    end
end
clear img_data
save('kclustCBP_Pal_voxval_subjs.mat','-v7.3');


M = 10;
parfor (i = 1:length(subjlist), M)
    fprintf('### calculating subj silhs: [%d / %d] ###\n',i,length(subjlist));
    
    r2z = atanh(corr(voxroi{i},voxWB{i},'Type','Pearson'));    % Fisher z transformation
    r2ze = r2z(:,all(~isnan(r2z)));
    
    idx{i} = zeros(length(ind{roi}),20); % temporary variables for parfor
    silh{i} = zeros(length(ind{roi}),20);
    cs{i} = zeros(1,20);
    
    clustMethod = 2;    % kmeans - 1; hclust - 2
    dist = 'euclidean';
    compalg = 'average';
    nmaxclust = 20; % refer to Balsters et al., 2018, Neuroimage
    if clustMethod == 1
        % k-means clustering with averaged silhouette
        repeat = 100;
        for s = 2:nmaxclust
            [idx{i}(:,s),C{i}(:,s),sumdist{i}(:,s)] = kmeans(r2ze,s,'Distance','sqeuclidean','Display','final','Replicates',repeat);

            % silhouette plot
            silh{i}(:,s) = silhouette(r2ze,idx{i}(:,s),dist);
            cs{i}(s) = mean(silh{i}(:,s));
        end
    else
        % hierarchical clustering with average linkage
        Y = pdist(r2ze,dist);
        Z = linkage(Y,compalg);
        dendrogram(Z); % visually check dendrogram of the clusters divided   
        for s = 2:nmaxclust
            idx{i}(:,s) = cluster(Z,'MaxClust',s);

            % silhouette plot
            silh{i}(:,s) = silhouette(r2ze,idx{i}(:,s),dist);
            cs{i}(s) = mean(silh{i}(:,s));
        end
    end
    parsave(sprintf('idxSubjs_%d.mat', i), idx{i});
    parsave(sprintf('silhSubjs_%d.mat', i), silh{i});
    parsave(sprintf('csSubjs_%d.mat', i), cs{i});
end
clear voxroi voxWB
save('kclustCBP_Pal_hclust_euc_avg_subjs.mat','-v7.3');
