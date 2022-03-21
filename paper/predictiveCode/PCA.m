function [W,L] = PCA(X,D,FLAG)
% June Kwon
%#ok<*AGROW>
COV = cov(X);
[EVEC, EVAL] = eig(COV);
if FLAG == 1 % If 1, Whiten
    EVEC = EVEC ./ sqrt(max(EVAL));
end
[EVAL_MAT, ~] = max(EVAL);

    %Extract Eigen Values / Vectors
    for i = 1 : D
        [MX_EVAL, MX_EIND] = max(EVAL_MAT);
        EVAL_MAT(MX_EIND) = [];
        MX_EVEC = EVEC(:,MX_EIND);
        % Store
        if (i == 1)
            W = MX_EVEC;
            L = MX_EVAL;
        else
            W = [W MX_EVEC];
            L = [L MX_EVAL];
        end     
    end
end