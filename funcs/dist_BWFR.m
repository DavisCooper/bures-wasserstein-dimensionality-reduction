function dBWFR = dist_BWFR(Set1,Set2)
dist_THRSH = 1e-6;
l1 = size(Set1,3);
n = size(Set1,1);

if (nargin < 2)
    dBWFR = zeros(l1);
    for tmpC1 = 1:l1
        X = Set1(:,:,tmpC1);
        for tmpC2 = tmpC1+1:l1
            [U,~,V] = svd(Set1(:,:,tmpC2)'*X);
            W = U*V';
            dBWFR(tmpC2,tmpC1) = norm(X - Set1(:,:,tmpC2)*W,'fro')^2;
            if  (dBWFR(tmpC2,tmpC1) < dist_THRSH)
                dBWFR(tmpC2,tmpC1) = 0.0;
            end
            dBWFR(tmpC1,tmpC2) = dBWFR(tmpC2,tmpC1);
        end
    end
    
    
else
    l2 = size(Set2,3);
    dBWFR = zeros(l2,l1);
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            [U,~,V] = svd(Set2(:,:,tmpC2)'*Set1(:,:,tmpC1));
            W = U*V';
            dBWFR(tmpC2,tmpC1) = norm(Set1(:,:,tmpC1) - Set2(:,:,tmpC2)*W,'fro')^2;
            if  (dBWFR(tmpC2,tmpC1) < dist_THRSH)
                dBWFR(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end
