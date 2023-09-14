function Cost = compute_node_cost(ProbFeatA,GallFeatB,P)

[FeatMat1, Count1] = feature_cell2mat(ProbFeatA); %cell2mat
[FeatMat2, Count2] = feature_cell2mat(GallFeatB); %cell2mat

distanceAB = EuclidDist(FeatMat1' * P, FeatMat2'* P); 

index1 = count2index(Count1);
index2 = count2index(Count2);

Cost_set   = zeros(length(Count1),length(Count2));

    % compute sequence cost
    for i = 1:length(Count1)
        for j = 1:length(Count2)
            tempDist      = distanceAB(index1(i,1):index1(i,2),index2(j,1):index2(j,2));   
            Cost_set(i,j) = mean(mean(tempDist)) ;  %mean
        end
    end
Cost = Cost_set;
end
function index = count2index(NumCount)
    % count to label index
    index =zeros(length(NumCount),2);   
    index(1,1) = 1;
    index(1,2) = NumCount(1);
    for i=2:length(NumCount)
        index(i,2) = sum(NumCount(1:i));
        index(i,1) = index(i-1,2)+1;
    end
end