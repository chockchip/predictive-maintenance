function [TREE] = MultiDecisionTree(XY,ATBT_Cell,ATBT_Active)
	% June Kwon
    % (XY   : Example)
    % (ATBT : Attribute)
    % (DEFT : Default)
    %#ok<*AGROW>
    %#ok<*NASGU>

    % Constants
    ATBT_Num = length(ATBT_Active);
    OBSV = size(XY,1);
    XT_BY = XY(:,1:end-1);
    YT = XY(:,end);
    UNIQ = unique(YT)';
    
    % Create the tree node
    TREE = struct('VALUE', 'null', 'LEFT', 'null', 'RIGHT', 'null');

    % If same class on all XY, return that class
    if length(UNIQ) == 1
        TREE.VALUE = num2str(UNIQ); %sprintf('%i',UNIQ);
        return
    end

    % If ATBT is empty, return mode(XY)
    if (sum(ATBT_Active) == 0)
        TREE.VALUE = num2str(mode(YT)); %sprintf('%i',mode(YT));
        return
    end

    % Else, Choose the best attribute
    % First, evaluate the entropy of target
    K = length(UNIQ);
    logb = @(x,b) log(x)/log(b);
    PY = 0;
    HY = 0;
    for i = unique(YT)'
        PY = size(YT(YT == i),1)/size(YT,1);
        S = logb(PY,K); S(isnan(S)) = 0;
        HY = HY - PY * logb(PY,K);
    end
    
    % Next, compute the information gain to find the best attribute
    GAIN = -1*ones(1,ATBT_Num); % -1 if inactive, 1 if active 
    % Loop through attributes to update only the active gains
    for i= 1 : ATBT_Num
        if (ATBT_Active(i)) % If active, update its gain
            S0 = 0; S0_T = 0;
            S1 = 0; S1_T = 0;
            for j = 1 : OBSV
                if (XY(j,i))
                    S1 = S1 + 1;
                    if (XY(j,ATBT_Num+1)) % if target attribute is true
                        S1_T = S1_T + 1;
                    end
                else
                    S0 = S0 + 1;
                    if (XY(j,ATBT_Num+1)) % if target attribute is true
                        S0_T = S0_T + 1;
                    end
                end
            end

            % Entropy for S(V=1)
            if (~S1)
                P1 = 0;
            else
                P1 = (S1_T / S1); 
            end

            if (P1 == 0)
                P1_Q = 0;
            else
                P1_Q = -1*(P1)*log2(P1);
            end

            if (~S1)
                P0 = 0;
            else
                P0 = ((S1 - S1_T)/S1);
            end

            if (P0 == 0)
                P0_Q = 0;
            else
                P0_Q = -1*(P0)*log2(P0);
            end

            HX_1 = P1_Q + P0_Q;

            % Entropy for S(v=0)
            if (~S0)
                P1 = 0;
            else
                P1 = (S0_T / S0); 
            end

            if (P1 == 0)
                P1_Q = 0;
            else
                P1_Q = -1*(P1)*log2(P1);
            end

            if (~S0)
                P0 = 0;
            else
                P0 = ((S0 - S0_T) / S0);
            end

            if (P0 == 0)
                P0_Q = 0;
            else
                P0_Q = -1*(P0)*log2(P0);
            end

            HX_0 = P1_Q + P0_Q;

            GAIN(i) = HY - ((S1/OBSV)*HX_1) - ((S0/OBSV)*HX_0);
        end
    end

    % Choose the attribute whose weighed average entropy is the lowest 
    [~, BEST] = max(GAIN);
    
    % Set tree.VALUE to BEST's relevant string
    TREE.VALUE = ATBT_Cell{BEST};
    
    % Remove splitting attribute from ATBT_Active
    ATBT_Active(BEST) = 0;

    % Create the new example matrices by splitting
    XY_1 = [];    XY_1_INDEX = 1;
    XY_0 = [];    XY_0_INDEX = 1;

    for i = 1 : OBSV
        if (XY(i, BEST))
            XY_1(XY_1_INDEX,:) = XY(i,:);
            XY_1_INDEX = XY_1_INDEX + 1;
        else
            XY_0(XY_0_INDEX,:) = XY(i,:);
            XY_0_INDEX = XY_0_INDEX + 1;
        end
    end
    
    % This is where the "default" is applied.
    
    % For VALUE = 0 (False), it corresponds to the left branch
    % If XY_0 is empty, add leaf node to the left with the default, which
    % in this case, default = mode(YT)
    if (isempty(XY_0))
        LEAF = struct('VALUE', 'null', 'LEFT', 'null', 'RIGHT', 'null');
        LEAF.VALUE = num2str(mode(YT)); %sprintf('%i',mode(YT));
        TREE.LEFT = LEAF;
    else
        % Now, Recur
        TREE.LEFT = MultiDecisionTree(XY_0,ATBT_Cell,ATBT_Active);
    end
    
    % For VALUE = 1 (True), it corresponds to the right branch
    % If XY_1 is empty, add leaf node to the right with the default, which
    % in this case, default = mode(YT)
    if (isempty(XY_1))
        LEAF = struct('VALUE', 'null', 'LEFT', 'null', 'RIGHT', 'null');
        LEAF.VALUE = num2str(mode(YT)); %sprintf('%i',mode(YT));
        TREE.RIGHT = LEAF;
    else
        % Now, Recur
        TREE.RIGHT = MultiDecisionTree(XY_1,ATBT_Cell,ATBT_Active);
    end

    % Now, return tree
    return

end

