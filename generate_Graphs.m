function G = generate_Graphs(inData,k_w,k_b,tmpDist)
% function [G_w,G_b] = generate_Graphs(varargin)

nPoints = length(inData.trn_y);
if (nargin < 2)
    error('the k should be defined');
elseif (nargin <3)
    k_b = k_w;    
elseif (nargin < 4)
    switch (inData.Metric_Flag)
        case 2
            %Stein metric
            tmpDist = Compute_Stein_Metric(inData.trn_X);
        case 1
            %AIRM metric
            tmpDist = Compute_AIRM_Metric(inData.trn_X);
        case 3
            %KLDM metric
            tmpDist = Compute_J_Metric(inData.trn_X);
        otherwise
            error('The metric is not implemented.');
    end %end switch
end

    
%Within Graph
G_w = zeros(nPoints);
for tmpC1 = 1:nPoints
    tmpIndex = find(inData.trn_y == inData.trn_y(tmpC1));
    [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
    if (length(tmpIndex) < k_w)
        max_w = length(tmpIndex);
    else
        max_w = k_w;
    end
    G_w(tmpC1,tmpIndex(sortInx(1:max_w))) = 1;
end

%Between Graph

G_b = zeros(nPoints);
for tmpC1 = 1:nPoints
    tmpIndex = find(inData.trn_y ~= inData.trn_y(tmpC1));
    [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
    if (length(tmpIndex) < k_b)
        max_b = length(tmpIndex);
    else
        max_b = k_b;
    end
    G_b(tmpC1,tmpIndex(sortInx(1:max_b))) = 1;
end

G = G_w - inData.lambda*G_b;
end


