function [crr,pred_y] = SPD_NN_Classifier(trnX,trnY,tstX,tstY,W,SPD_metric)
if (nargin < 6)
    SPD_metric = 1;
end

p         = size(W,2);
l1        = size(trnX,3);
WTtrnXW   = zeros(p,p,l1);
for tmpC1 = 1:l1
    WTtrnXW(:,:,tmpC1) = W'*trnX(:,:,tmpC1)*W;
end

l1        = size(tstX,3);
WTtstXW   = zeros(p,p,l1);
for tmpC1 = 1:l1
    WTtstXW(:,:,tmpC1) = W'*tstX(:,:,tmpC1)*W;
end

switch(SPD_metric)
    case 1  %AIRM
        dist_mat = dist_AIRM(WTtrnXW,WTtstXW);
    case 2  %Stein
        dist_mat = dist_Stein(WTtrnXW,WTtstXW);
    case 3  %Jeffrey
        dist_mat = dist_Jeffrey(WTtrnXW,WTtstXW);
    case 4  %log-Euclidean
        dist_mat = dist_logEuc(WTtrnXW,WTtstXW);
    case 5  %Euclidean
        dist_mat = dist_Euc(WTtrnXW,WTtstXW);
    otherwise
        error('The metric is not defined');
end

[~,tmpIdx] = min(dist_mat,[],2);
pred_y     = trnY(tmpIdx);
crr        = sum(pred_y(:) == tstY(:))/l1;
end