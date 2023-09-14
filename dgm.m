addpath ('lib/','data/');

%% 
dataset_name = 'prid'; % 'prid' or 'ilids'
pcadim       = 600;
nTrial       = 10;

%% 
disp(['Evaluation on ' dataset_name ' dataset.']);
switch dataset_name
    case 'prid'
        load('splits_prid.mat'); 
        load('PRID_LOMO_PCA_feature.mat');
    case 'ilids'
        load('splits_ilidsvid.mat');
        load('ILIDS_LOMO_PCA_feature.mat');
end

Cam_Feat_1 = Cam_Feat_1(:,1:pcadim);
Cam_Feat_2 = Cam_Feat_2(:,1:pcadim);

% Matrix transforms to cells for each person
cam1_feat = feature_mat2cell(Cam_Feat_1',count1);
cam2_feat = feature_mat2cell(Cam_Feat_2',count2);

% Regularized nearest points set to set distance (higher rank-1)
CMC_set  = zeros(nTrial,size(ls_set,2)/2); 
% Average set to set distance (higher AUC)
CMC_mean = zeros(nTrial,size(ls_set,2)/2); 

for trial  = 1:nTrial
    
disp(['Experiment No.',num2str(trial) '`.......']) 

%Generate training/testing idx
test_split = ls_set(trial,:);

TrainIdx = test_split(1:length(test_split)/2);
TestIdx  = test_split(length(test_split)/2+1:end);

TrainFeat1 = cam1_feat(TrainIdx);
TrainFeat2 = cam2_feat(TrainIdx);
TestFeat1 = cam1_feat(TestIdx);
TestFeat2 = cam2_feat(TestIdx);
[TrainFeatMat1, TrainCount1] = feature_cell2mat(TrainFeat1); %cell2mat
[TrainFeatMat2, TrainCount2] = feature_cell2mat(TrainFeat2);
%[CNNFeat, CNNCount1, CNNCount2] = feature_cell2mat1(TrainFeat1, TrainFeat2);
 %cell2mat

% Compute the original graph cost
P = eye(size(Cam_Feat_1,2));   
Graph_Cost = compute_node_cost(TrainFeat1,TrainFeat2,P);
Graph_Cost1 = compute_node_cost(TestFeat1,TestFeat2,P);
mu = mean(Graph_Cost(:));
mu1 = mean(Graph_Cost1(:));

% compute the center of each image sequence
probFea = zeros(length(TrainFeat1),size(TrainFeat1{1},1));
gallFea = zeros(length(TrainFeat1),size(TrainFeat1{1},1));
for i =1:length(TrainFeat1)
    tmpProbFea = mean(TrainFeat1{i}, 2);
    tmpGallFea = mean(TrainFeat2{i}, 2);
    probFea(i,:) = tmpProbFea';
    gallFea(i,:) = tmpGallFea';
end

% parameters for metric learning
tol = 1e-4;
L = 1 / 2^8;
gamma = 2;
prevAlpha = 0;
prevM  = P;
M      = P;
prevP2 = P;

for iter = 1:10
fprintf('Trial %d   Iter %d  ', trial, iter); 

%% Graph cost construction
Graph_Cost = Graph_Cost-mu;
Graph_Cost = log(1 + exp( Graph_Cost ));
Graph_Cost1 = Graph_Cost1-mu1;
Graph_Cost1 = log(1 + exp( Graph_Cost1 ));

% Graph matching
[X,score] = hungarian(Graph_Cost);
Train_label=X;
Yp        = label_reweighting(X,Graph_Cost); % with re-weighting
%Yp        = label_transform(X,Graph_Cost); % without re-weighting
[Q,score1] = hungarian(Graph_Cost1);

Yp1        = label_reweighting(Q,Graph_Cost1);

F1 = score/(size(X,1)*mean(mean(Graph_Cost)));

%% update the metric
nPos = length(find(Yp(:)>0.5));
nNeg = sum(Yp(:) == -1);
Wp = zeros(length(TrainFeat1), length(TrainFeat1));
Wp(Yp > 0.5) = 1 / nPos;
Wp(Yp == -1) = 1 / nNeg;
WYp = Wp .* Yp;

Dp = Graph_Cost - mu;
Dp(Yp == -1) = - Dp(Yp == -1);
Dp(Yp >0.5)  = Dp(Yp >0.5).*Yp(Yp >0.5);
newF = log(1 + exp( Dp )); % log(1 + exp( D ));
newF = Wp(:)' * newF(:); % sum(sum( W .* log(1 + exp( Y .* (D - mu) )) ));

for r = 1:50
newAlpha = (1 + sqrt(1 + 4 * prevAlpha^2)) / 2;
V = M + (prevAlpha - 1) / newAlpha * (M - prevM);
alpha = -(prevAlpha - 1) / newAlpha; % for prevP1
beta = 1 + (prevAlpha - 1) / newAlpha; % for prevP2

prevP1 = prevP2;
prevP2 = P;
prevM = M;
prevF = newF;
prevAlpha = newAlpha;

Dp = alpha * EuclidDist(probFea * prevP1, gallFea * prevP1) + beta * EuclidDist(probFea * prevP2, gallFea * prevP2) - mu;

Dp(Yp == -1) = - Dp(Yp == -1);
Dp(Yp >0.5)  = Dp(Yp >0.5).*Yp(Yp >0.5);

T = WYp ./ (1 + exp( -Dp ));
X = probFea' * T * gallFea;
gradF = probFea' * bsxfun(@times, sum(T, 2), probFea) - X - X' + bsxfun(@times, gallFea', sum(T, 1)) * gallFea;

prevF_V =  log(1 + exp( Dp ));
prevF_V = Wp(:)' * prevF_V(:);

while true
    [optFlag, M, P, latent, r, newF] = LineSearch(V, gradF, prevF_V, probFea, gallFea, Yp, Wp, L, mu);
    if ~optFlag
        L = gamma * L;
    else
        break;
    end
end
end
F2 = newF;

fprintf('ObF %2.4f\n', F1 + F2);
% update graph
Graph_Cost = compute_node_cost(TrainFeat1,TrainFeat2,P);
Graph_Cost1 = compute_node_cost(TestFeat1,TestFeat2,P);

end



[TestFeatMat1, TestCount1] = feature_cell2mat(TestFeat1); %cell2mat
[TestFeatMat2, TestCount2] = feature_cell2mat(TestFeat2); %cell2mat
%[CNNTestFeat, CNNTestCount1, CNNTestCount2] = feature_cell2mat1(TestFeat1, TestFeat2);
% compute mean set-to-set distance
dist_eucl = EuclidDist(TestFeatMat1' * P, TestFeatMat2'* P); 
mean_dist = compute_set_distance(dist_eucl,TestCount1,TestCount2);
CMC_mean(trial,:) = calc_CMC(mean_dist);

% regularized nearest points (it takes a longer time)
set_dist = learn_set_dist(TestFeat1,TestFeat2, P);
CMC_set(trial,:)  =  calc_CMC(set_dist);

end

cmcp(1,:) = mean(CMC_mean,1);
cmcp(2,:) = mean(CMC_set,1);
fprintf('.......................The Average Performance.........................\n');
fprintf('..................Rank1, Rank5, Rank10, Rank15, Rank20..................\n');
fprintf('Average Distance: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%\n', (cmcp(1,[1; 5; 10; 15; 20]))*100);
fprintf('    Set Distance: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%\n', (cmcp(2,[1; 5; 10; 15; 20]))*100);
