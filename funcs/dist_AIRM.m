function dAIRM = dist_AIRM(Set1,Set2)
dist_THRSH = 1e-6;
l1 = size(Set1,3);
n = size(Set1,1);
I_n = eye(n);
if (nargin < 2)
    dAIRM = zeros(l1);
    for tmpC1 = 1:l1
        iX = I_n/Set1(:,:,tmpC1);
        for tmpC2 = tmpC1+1:l1
            dAIRM(tmpC2,tmpC1) = real(trace((logm(Set1(:,:,tmpC2)*iX))^2));
            if  (dAIRM(tmpC2,tmpC1) < dist_THRSH)
                dAIRM(tmpC2,tmpC1) = 0.0;
            end
            dAIRM(tmpC1,tmpC2) = dAIRM(tmpC2,tmpC1);
        end
    end
    
    
else
    l2 = size(Set2,3);
    dAIRM = zeros(l2,l1);
    for tmpC1 = 1:l1
        iX = I_n/Set1(:,:,tmpC1);
        for tmpC2 = 1:l2
            dAIRM(tmpC2,tmpC1) = real(trace((logm(Set2(:,:,tmpC2)*iX))^2));
            if  (dAIRM(tmpC2,tmpC1) < dist_THRSH)
                dAIRM(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end
