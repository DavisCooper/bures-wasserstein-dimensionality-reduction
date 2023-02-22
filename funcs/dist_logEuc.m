function dlogEuc = dist_logEuc(Set1,Set2)
dist_THRSH = 1e-6;
n = size(Set1,1);
l1 = size(Set1,3);
if (nargin < 2)
    tmpMat = zeros(n*(n+1)/2,l1);
    for tmpC1 = 1:l1
        tmpMat(:,tmpC1) = SPD2Euclidean(logm(Set1(:,:,tmpC1)));
    end
    dlogEuc = distEucVec( tmpMat', tmpMat')';
    
else
    l2 = size(Set2,3);
    tmpMatX = zeros(n*(n+1)/2,l1);
    for tmpC1 = 1:l1
        tmpMatX(:,tmpC1) = SPD2Euclidean(logm(Set1(:,:,tmpC1)));
    end
    tmpMatY = zeros(n*(n+1)/2,l2);
    for tmpC1 = 1:l2
        tmpMatY(:,tmpC1) = SPD2Euclidean(logm(Set2(:,:,tmpC1)));
    end
    dlogEuc = distEucVec( tmpMatX', tmpMatY')';
    
end
dlogEuc(dlogEuc < dist_THRSH) = 0;

end

%---------------
function D = distEucVec( X, Y )
Yt = Y';
XX = sum(X.*X,2);
YY = sum(Yt.*Yt,1);
D = bsxfun(@plus,XX,YY) - 2*X*Yt;
end
