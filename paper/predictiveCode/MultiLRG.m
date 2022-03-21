function W = MultiLRG(XT,YT,i,j)

	% June Kwon
    %#ok<*AGROW>
    [N,D] = size(XT);                  % Number of Observation & Feature
    R = [-1 1];                      % Range of Random Numbers
    w = rand(D,1)*range(R) + min(R);   % Initial Guessed Parameters
    LR = 0.1;                          % Learning Rate

    FLAG = true;                       % Flag
    COUNT = 1;                         % Count
    LIMIT = 20000;                     % Limit of Iteration
    PC = 1.0e-7;                       % Limit of Percent Change

    NT = size(XT,1);
    MNLOG_TRN_MAT = [];

    while (COUNT < LIMIT) && FLAG

        % Previous Parameter to compute difference
        w0 = w;
        YH_Training_0 = XT*w0;
        MNLOG_TRN_Update_0 = (1/NT)*(YT'*log(YH_Training_0)+(1-YT)'*log(1-YH_Training_0));
        MNLOG_TRN_Update_0 = real(MNLOG_TRN_Update_0);

        GRAD = (1/N)*XT'*(YT-(XT*w));  % Gradient
        w = w + LR*GRAD;               % Updating the gradient

        % Compute mean log likelihood 
        YH_Training = XT*w;            % Predicted Data on Training data
        for k = 1:length(YH_Training)
            if YH_Training(i) == 0
                YH_Training(i) = 1e-5;
            end
        end
        MNLOG_TRN_Update = (1/NT)*(YT'*log(YH_Training)+(1-YT)'*log(1-YH_Training));
        MNLOG_TRN_Update = real(MNLOG_TRN_Update);
        MNLOG_TRN_MAT = [MNLOG_TRN_MAT ; MNLOG_TRN_Update];

        d = abs(MNLOG_TRN_Update - MNLOG_TRN_Update_0);

        % Condition Check if distance is less than percent change
        if (d < PC)
            FLAG = false;
        end

        COUNT = COUNT + 1;             % Increment

    end
   
    % [6]. Figure
    % figure; plot(MNLOG_TRN_MAT,'linewidth',2);
    % xlabel('Number of Epoch'); ylabel('Mean Log Likelihood');
    % legend("Training"); grid on;
    % title(sprintf('Logistic Regression on Class %d and Class %d',i,j));
    
    W = w;

end