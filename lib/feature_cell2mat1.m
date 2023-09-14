function [feature_mat, CountOfSample1, CountOfSample2]= feature_cell2mat1(feature_cell1,feature_cell2)
% input:
%   feature_cell :N*1 cell N is the nmuber of samples
    
% output:
%   feature_mat
%	CountOfSample: The number of each sample

feature_mat   = [];
feature_mat1  =  [];
CountOfSample1 = zeros(1,length(feature_cell1));
CountOfSample2 = zeros(1,length(feature_cell2));

for iPerson = 1:length(feature_cell1)
    temp_feature1 = feature_cell1{iPerson};
    CountOfSample1(iPerson) = size(temp_feature1,2);
    for jPerson = 1:length(feature_cell2)
        temp_feature2 = feature_cell2{jPerson};
        CountOfSample2(jPerson) = size(temp_feature2,2);
        feature_mat1 = [feature_mat1 temp_feature1];
        feature_mat1 = [feature_mat1 temp_feature2];
    
    numOfColumns =  40 - (CountOfSample1(iPerson)+CountOfSample2(jPerson));
    if(numOfColumns ~= 0)
        Z = zeros(size(feature_mat1, 1), numOfColumns);
        feature_mat1 = horzcat(feature_mat1, Z);
    end
    end
    feature_mat = [feature_mat' feature_mat1'];
    feature_mat=feature_mat';    
        feature_mat1=  [];
    %print(feature_mat)
    %print(size(feature_mat))
end

end