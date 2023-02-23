function [trn_chol_X] = vec_chol(trn_X)
    [n,~,m] = size(trn_X);
    trn_chol_X = zeros(n,n,m);
    for i=1:m
        trn_chol_X(:,:,i) = chol(trn_x(:,:,i))';
    end

end