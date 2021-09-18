% prep data
grp = ['OCD'];   %HCs, OCD, or all
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
for g = 1:length(maskROIs) % striatum, pallidum, thalamus
    mask{g} = load_nii([dataDir '/HO_atlas_thr50/HO_' maskROIs{g} '_2mm_bin.nii']);
    maskCoord{g} = mask{g}.img(:,:,:) ~= 0;     % extract non-zero values
    ind{g} = find(maskCoord{g});
    sz{g} = size(maskCoord{g});
    [x{g},y{g},z{g}] = ind2sub(sz{g},ind{g});   % assign mask coordinates of each ROI
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
roi = 3; % striatum - 1; pallidum - 2; thalamus - 3
for i = 1:length(subjlist)
    for k = 1:length(ind{roi})
        voxroi{i}(1:112,k) = img_data{i}(x{roi}(k),y{roi}(k),z{roi}(k),1:112);
    end
    for kk = 1:length(ind_wb{roi})
        voxWB{i}(1:112,kk) = img_data{i}(x_wb{roi}(kk),y_wb{roi}(kk),z_wb{roi}(kk),1:112);
    end
end
clear img_data

for k = 1:length(ind{roi})
    for i = 1:length(subjlist)
        r2z(i,:) = atanh(corr(voxroi{i}(:,k),voxWB{i},'Type','Pearson'));   % Fisher z transformation
    end
    mr2z(k,:) = mean(r2z,1);
    clear r2z
end
clear voxroi voxWB

mr2ze = mr2z(:,all(~isnan(mr2z)));  % excluding whole-brain voxels with NaN intensities
save('kclustCBP_Thal_OCD.mat','-v7.3');


%% clustering on the averaged connmat
clear idx C sumdist Y Z silh h cs
clustMethod = 2; % kmeans - 1; hclust - 2
dist = 'euclidean';
compalg = 'average';
nmaxclust = 20; % refer to Balsters et al., 2018, Neuroimage
if clustMethod == 1
    % k-means clustering with averaged silhouette
    repeat = 100;
    for s = 2:nmaxclust
        [idx{s},C{s},sumdist{s}] = kmeans(mr2ze,s,'Distance','sqeuclidean','Display','final','Replicates',repeat);

        % silhouette plot
        [silh{s},h{s}] = silhouette(mr2ze,idx{s},dist);
        cs(s) = mean(silh{s});
    end
else
    % hierarchical clustering with average linkage
    Y = pdist(mr2ze,dist);
    Z = linkage(Y,compalg);
%    dendrogram(Z);    % visually check dendrogram of the clusters divided   
    for s = 2:nmaxclust
        idx{s} = cluster(Z,'MaxClust',s);

        % silhouette plot
        [silh{s},h{s}] = silhouette(mr2ze,idx{s},dist);
        cs(s) = mean(silh{s});
    end
end
save('kclustCBP_Thal_hclust_euc_avg_OCD.mat','-v7.3');
