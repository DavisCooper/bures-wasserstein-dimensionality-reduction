function dStein = dist_Stein(Set1,Set2)
dist_THRSH = 1e-6;
l1 = size(Set1,3);
LOGDET_X = zeros(1,l1);
for tmpC1 = 1:l1
    LOGDET_X(tmpC1) = logdet(Set1(:,:,tmpC1),'chol');
end
if (nargin < 2)
    dStein = zeros(l1);
    for tmpC1 = 1:l1
        for tmpC2 = tmpC1+1:l1
            dStein(tmpC2,tmpC1) = logdet(0.5*(Set1(:,:,tmpC1) + Set1(:,:,tmpC2)),'chol') -  0.5*LOGDET_X(tmpC2) -0.5*LOGDET_X(tmpC1);
            if  (dStein(tmpC2,tmpC1) < dist_THRSH)
                dStein(tmpC2,tmpC1) = 0.0;
            end
            dStein(tmpC1,tmpC2) = dStein(tmpC2,tmpC1);
        end
    end
    
    
else
    l2 = size(Set2,3);
    dStein = zeros(l2,l1);
    LOGDET_Y = zeros(1,l2);
    for tmpC1 = 1:l2
        LOGDET_Y(tmpC1) = logdet(Set2(:,:,tmpC1),'chol');
    end
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            dStein(tmpC2,tmpC1) = logdet(0.5*(Set1(:,:,tmpC1) + Set2(:,:,tmpC2)),'chol') -  0.5*LOGDET_X(tmpC1) -0.5*LOGDET_Y(tmpC2);
            if  (dStein(tmpC2,tmpC1) < dist_THRSH)
                dStein(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end