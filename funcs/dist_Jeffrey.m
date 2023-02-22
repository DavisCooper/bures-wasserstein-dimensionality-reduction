function dJeffrey = dist_Jeffrey(Set1,Set2)
dist_THRSH = 1e-6;
n = size(Set1,1);
I_n = eye(n);
l1 = size(Set1,3);
iX = zeros(size(Set1));
for tmpC1 = 1:l1
    iX(:,:,tmpC1) = I_n/Set1(:,:,tmpC1);
end
if (nargin < 2)
    dJeffrey = zeros(l1);
    for tmpC1 = 1:l1
        for tmpC2 = tmpC1+1:l1
            dJeffrey(tmpC2,tmpC1) = 0.5*trace(iX(:,:,tmpC1)*Set1(:,:,tmpC2) + iX(:,:,tmpC2)*Set1(:,:,tmpC1))- n;
            if  (dJeffrey(tmpC2,tmpC1) < dist_THRSH)
                dJeffrey(tmpC2,tmpC1) = 0.0;
            end
            dJeffrey(tmpC1,tmpC2) = dJeffrey(tmpC2,tmpC1);
        end
    end
else
    l2 = size(Set2,3);
    dJeffrey = zeros(l2,l1);
    
    iY = zeros(size(Set2));
    for tmpC1 = 1:l2
        iY(:,:,tmpC1) = I_n/Set2(:,:,tmpC1);
    end
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            dJeffrey(tmpC2,tmpC1) = 0.5*trace(iX(:,:,tmpC1)*Set2(:,:,tmpC2) + iY(:,:,tmpC2)*Set1(:,:,tmpC1))- n;
            if  (dJeffrey(tmpC2,tmpC1) < dist_THRSH)
                dJeffrey(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end