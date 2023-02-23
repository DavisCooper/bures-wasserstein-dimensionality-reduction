classdef spdDR < handle
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    properties
        %general parameters
        metric = 1; %1: AIRM, 2: Stein, 3: Jeffrey, 4: logEuclidean, 5: Euclidean
        nIter = 10;
        trn_X = [];
        trn_y = [];
        
        newDim = [];
        
        k_w = 3;    %within graph neighbor size for discriminant analysis
        k_b = 1;    %between graph neighbor size for discriminant analysis
        graph_lambda = 1;
        
        verbose = true;
    end
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    properties (Access = private)
        nTrn = [];
        origDim = [];
        nClasses = [];        
        log_trn_X = [];
        chol_trn_X = [];
        aff_graph = [];
    end
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    methods
        function W = perform_graph_DA(obj,metric)
            if (nargin == 2)
                obj.metric = metric;
            end
            obj.origDim = size(obj.trn_X,1);
            obj.nTrn = size(obj.trn_X,3);
            obj.nClasses = max(obj.trn_y);
            
            switch(obj.metric)
                case 1  %AIRM
                    W = obj.perform_AIRM_DA();
                case 2  %Stein
                    W = obj.perform_Stein_DA();
                case 3  %Jeffrey
                    W = obj.perform_Jeffrey_DA();
                case 4  %log-Euclidean
                    W = obj.perform_log_euc_DA();
                case 5  %Euclidean
                    W = obj.perform_euc_DA();
                case 6  %Bures-Wasserstein
                    W = obj.perform_BW_DA();
                case 7  %Fixed-Rank Bures-Wasserstein
                    W = obj.perform_BWFR_DA();
                otherwise
                    error('The metric is not defined');
            end
            
        end
    end
    
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    methods(Access = private)
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_AIRM_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using AIRM metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_AIRM(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_AIRM(obj,W);
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
      
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_Stein_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Stein metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_Stein(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_Stein(obj,W);
            %             checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_Jeffrey_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Jeffrey metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_Jeffrey(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_Jeffrey(obj,W);
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end       
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_log_euc_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using log-Euclidean metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            if (isempty(obj.log_trn_X))
                if (obj.verbose)
                    fprintf('Preparing intermediate data.\n');
                end
                
                obj.log_trn_X = zeros(obj.origDim,obj.origDim,obj.nTrn);
                for tmpC1 = 1:obj.nTrn
                    obj.log_trn_X(:,:,tmpC1) = logm(obj.trn_X(:,:,tmpC1));
                end
            end
            
            %generating the affinity function
            euc_dist_orig = dist_logEuc(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(euc_dist_orig);
            
            %initializing
            W = eye(obj.origDim,obj.newDim);
            for tmpIter = 1:obj.nIter
                
                F_W = obj.compute_F_W_log_Euc(obj.aff_graph,W);
                if (obj.verbose)
                    %computing the cost
                    cost0 = trace(W'*F_W*W);
                end
                
                [W,~] = eigs(F_W,obj.newDim,'sa');
                %computing the cost
                if (obj.verbose)
                    cost1 = trace(W'*F_W*W);
                    fprintf('iter%d. Cost function before and after update %.3f -> %.3f.\n',tmpIter,cost0,cost1);
                end
            end %endfor iteration
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_euc_DA(obj)
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Euclidean metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            le_dist_orig  = dist_Euc(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(le_dist_orig);
            
            %initializing
            W = eye(obj.origDim,obj.newDim);
            for tmpIter = 1:obj.nIter
                
                F_W = obj.compute_F_W_Euc(obj.aff_graph,W);
                if (obj.verbose)
                    %computing the cost
                    cost0 = trace(W'*F_W*W);
                end
                
                [W,~] = eigs(F_W,obj.newDim,'sa');
                %computing the cost
                if (obj.verbose)
                    cost1 = trace(W'*F_W*W);
                    fprintf('iter%d. Cost function before and after update %.3f -> %.3f.\n',tmpIter,cost0,cost1);
                end
            end %endfor iteration
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
        end

        function W = perform_BW_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using AIRM metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_BW(obj.trn_X); % IMPLEMENT
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_BW(obj,W); %IMPLEMENT
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end

        function W = perform_BWFR_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using AIRM metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            if (isempty(obj.chol_trn_X))
                if (obj.verbose)
                    fprintf('Preparing intermediate data.\n');
                end
                
                obj.chol_trn_X = vec_chol(ob.trn_X);
                
            end

            %generating the affinity function
            dist_orig = dist_BW(obj.chol_trn_X); % IMPLEMENT
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_BWFR(obj,W); %IMPLEMENT
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function a = generate_Graphs(obj,tmpDist)
            %Within Graph
            G_w = zeros(obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                tmpIndex = find(obj.trn_y == obj.trn_y(tmpC1));
                [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
                if (length(tmpIndex) < obj.k_w + 1)
                    max_w = length(tmpIndex);
                else
                    max_w = obj.k_w + 1;
                end
                G_w(tmpC1,tmpIndex(sortInx(1:max_w))) = 1;
            end
            %Between Graph
            G_b = zeros(obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                tmpIndex = find(obj.trn_y ~= obj.trn_y(tmpC1));
                [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
                if (length(tmpIndex) < obj.k_b)
                    max_b = length(tmpIndex);
                else
                    max_b = obj.k_b;
                end
                G_b(tmpC1,tmpIndex(sortInx(1:max_b))) = 1;
            end
            
            a = G_w - obj.graph_lambda*G_b;
        end
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_AIRM(obj,W)
            I_m = eye(obj.newDim);
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
                iWXW(:,:,tmpC1) = I_m/WXW(:,:,tmpC1);
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_AIRM(WXW(:,:,i(tmpC1)) , WXW(:,:,j(tmpC1)));
                X_i = obj.trn_X(:,:,i(tmpC1));
                X_j = obj.trn_X(:,:,j(tmpC1));
                
                log_XY = logm(WXW(:,:,i(tmpC1))*iWXW(:,:,j(tmpC1)));              
                dF = dF + 4*a_ij(tmpC1)*((X_i*W)*iWXW(:,:,i(tmpC1)) -(X_j*W)*iWXW(:,:,j(tmpC1)) )*log_XY;
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_Stein(obj,W)
            I_m = eye(obj.newDim);
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
                iWXW(:,:,tmpC1) = I_m/WXW(:,:,tmpC1);
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_Stein(WXW(:,:,i(tmpC1)) , WXW(:,:,j(tmpC1)));
                X_i = obj.trn_X(:,:,i(tmpC1));
                X_j = obj.trn_X(:,:,j(tmpC1));
                X_ij = 0.5*(X_i + X_j);
                dF = dF + a_ij(tmpC1)*(2*(X_ij*W)/(W'*X_ij*W)  ...
                    - (X_i*W)*iWXW(:,:,i(tmpC1)) - (X_j*W)*iWXW(:,:,j(tmpC1)));
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_Jeffrey(obj,W)
            I_m = eye(obj.newDim);
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
                iWXW(:,:,tmpC1) = I_m/WXW(:,:,tmpC1);
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_Jeffrey(WXW(:,:,i(tmpC1)) , WXW(:,:,j(tmpC1)));
                X_i = obj.trn_X(:,:,i(tmpC1));
                X_j = obj.trn_X(:,:,j(tmpC1));
                W_iWXW_i = W*iWXW(:,:,i(tmpC1));
                W_iWXW_j = W*iWXW(:,:,j(tmpC1));
                dF = dF + a_ij(tmpC1)*((X_j*W_iWXW_i - X_i*W_iWXW_i*WXW(:,:,j(tmpC1))*iWXW(:,:,i(tmpC1)) + ...
                    X_i*W_iWXW_j - X_j*W_iWXW_j*WXW(:,:,i(tmpC1))*iWXW(:,:,j(tmpC1))));
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end


        function [outCost,outGrad] = graph_DA_CostGrad_BW(obj,W)
            
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_BW(WXW(:,:,i(tmpC1)) , WXW(:,:,j(tmpC1)));
                X_i = obj.trn_X(:,:,i(tmpC1));
                X_j = obj.trn_X(:,:,j(tmpC1));

                XiW = Xii*W; WtXiW = W'*XiW;
                XjW = X_j*W; WtXjW = W'*XjW;
                
                f = f - trace(WtXiW + WtXjW) + 2*trace(real(sqrtm(WtXiW*WtXjW)));
                
                WtXiWhalf = real(sqrtm(WtXiW));
                
                R = real(dsqrtm(WtXiWhalf*WtXjW*YtXiiYhalf, eye(r)));
                dF = dF + 2*a_ij(tmpC1)* (XiW + XjW - 2*XjW*(WtXiWhalf *R*WtXiWhalf) - 2*X_iX_j*dsqrtm(WtXiiW, R*WtXiWhalf *WtXjW ) - 2*XiW* dsqrtm(WtXiW, WtXjW*WtXiWhalf*R));
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end

        function [outCost,outGrad] = graph_DA_CostGrad_BWFR(obj,W)
            chol_XW = zeros(obj.origDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                chol_XW(:,:,tmpC1) = obj.chol_trn_X(:,:,tmpC1)*W;
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_BWFR(chol_XW(:,:,i(tmpC1)) , chol_XW(:,:,j(tmpC1)));
                
                cholXi = obj.chol_trn_X(:,:,i(tmpC1));
                cholXj = obj.chol_trn_X(:,:,j(tmpC1));
                
                [U,~,V] = svd(W'*cholXj'*cholXi*W);
                W = U*V';
                dF = dF + 2*a_ij(tmpC1)*(cholXi'*cholXi*W - cholXj'*cholXi*W*U - cholXi'*cholXj*W*U' + cholXj'*cholXj*W);
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end

        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function WXW = map_trn_X(obj,W)
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
            end
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function F_W = compute_F_W_log_Euc(obj,a,W)
            F_W = zeros(obj.origDim,obj.origDim);
            a_sym = a + a';
            a_sym(1:obj.nTrn+1:end) = 0;
            [i,j,a_ij] = find(a_sym);
            WW = W*W';
            for tmpC1 = 1:length(i)
                diff = obj.log_trn_X(:,:,i(tmpC1)) - obj.log_trn_X(:,:,j(tmpC1));
                F_W = F_W + a_ij(tmpC1)*diff*WW*diff;
            end
            F_W = obj.symm(F_W);
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function F_W = compute_F_W_Euc(obj,a,W)
            F_W = zeros(obj.origDim,obj.origDim);
            a_sym = a + a';
            a_sym(1:obj.nTrn+1:end) = 0;
            [i,j,a_ij] = find(a_sym);
            WW = W*W';
            for tmpC1 = 1:length(i)
                diff = obj.trn_X(:,:,i(tmpC1)) - obj.trn_X(:,:,j(tmpC1));
                F_W = F_W + a_ij(tmpC1)*diff*WW*diff;
            end
            F_W = obj.symm(F_W);
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function D = distEucVec(~, X, Y )
            Yt = Y';
            XX = sum(X.*X,2);
            YY = sum(Yt.*Yt,1);
            D = bsxfun(@plus,XX,YY) - 2*X*Yt;
        end

        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function sym_X = symm(~,X)
            sym_X = .5*(X + X');
        end
        
    end
end