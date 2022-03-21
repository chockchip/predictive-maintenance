function YV_Hat = MultiNaiveBayes(XT,YT,XV,YV,THSH)
    % June Kwon
	%#ok<*FNDSB>
    
    % Reads in Multiclass Labels
    UNIQ = unique([YT ; YV])';
    PYX_MAT = []; %#ok<*AGROW>
    PY_MAT = [];

    for i = UNIQ
        % [Temporarily] Store Validation Data
        XV_IN = XV;
        
        % Split training data based on class label
        XT_MT = XT(find(YT == i),:);          % (MT : stands for Multi)
        
        % Compute Prior
        PY_MT = size(XT_MT,1) / (size(XT,1)); % (PY : stands for P(Y = i))
        PY_MAT = [PY_MAT PY_MT];
        
        % Compute Mean & Standard Deviation of Features
        MU_XT_MT = mean(XT_MT);               % (MU : stands for Mean)
        STD_XT_MT = std(XT_MT);               % (STD : stands for Standard Deviation)

        % [STD = 0 Cases] Remove the feature that has STD <= THSH
        if sum(STD_XT_MT <= THSH) >= 1        
            WRG0 = STD_XT_MT <= THSH;         % Usually, THSH = 0.0001
            C0 = find(WRG0 == 1);             % Find the index of STD <= THSH
            XV_IN(:,C0) = [];                 % Remove the feature in XV
            STD_XT_MT(C0) = [];               % Remove the feature in STD 
            MU_XT_MT(C0) = [];                % Remove the feature in MU
        end

        % Compute Normal PDF (Gaussian Distribution)
        PXY_MT = normpdf(XV_IN, MU_XT_MT, STD_XT_MT); % (PXY : stands for P(X|Y=i))
        PXY_MT_PROD = prod(PXY_MT, 2);                % Apply Naive Assumption
        
        % Create Naive Bayes Classification Model for Class i
        PYX_MT = (PY_MT .* PXY_MT_PROD);              % (PYX : stands for P(Y=i|X))

        % Store Resulting Probability for All Classes
        PYX_MAT = [PYX_MAT PYX_MT];                   % (MAT : stadns for Matrix)
    end

    % Classify Validation Data and Compute Predicted Y
    YV_Hat = zeros(size(PYX_MAT,1),1); % (YV_Hat : Predicted Y for Validation Data)
    for j = 1 : size(PYX_MAT, 1)
        % Reads in Each Row
        ROW = PYX_MAT(j,:);
        
        % if zero tie exists, assign the random class
        % else, assign the class whose P(Y=i|X) is the highest
        if sum(ROW) == 0
            YV_Hat(j) = randi([min(UNIQ) max(UNIQ)]);
        else
            [~,I] = max(ROW);
            
            if min(UNIQ) == 0
                I = I - 1;     % If handling binary
            end
            
            YV_Hat(j) = I;
        end
        
    end
    
end
