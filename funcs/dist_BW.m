function dBW = dist_BW(Set1,Set2)
dist_THRSH = 1e-6;
l1 = size(Set1,3);
n = size(Set1,1);

if (nargin < 2)
    dBW = zeros(l1);
    for tmpC1 = 1:l1
        for tmpC2 = tmpC1+1:l1
            dBW(tmpC2,tmpC1) = trace(Set1(:,:,tmpC1) + Set1(:,:,tmpC2) - 2*sqrtm(Set1(:,:,tmpC1)*Set1(:,:,tmpC2)));
            if  (dBW(tmpC2,tmpC1) < dist_THRSH)
                dBW(tmpC2,tmpC1) = 0.0;
            end
            dBW(tmpC1,tmpC2) = dBW(tmpC2,tmpC1);
        end
    end
    
    
else
    l2 = size(Set2,3);
    dBW = zeros(l2,l1);
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            dBW(tmpC2,tmpC1) = trace(Set1(:,:,tmpC1) + Set2(:,:,tmpC2) - 2*sqrtm(Set1(:,:,tmpC1)*Set2(:,:,tmpC2)));
            if  (dBW(tmpC2,tmpC1) < dist_THRSH)
                dBW(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end
