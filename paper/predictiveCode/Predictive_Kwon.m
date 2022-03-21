%% Project: Predictive Maintenance
% June Kwon
% 5/28/2021

clc
clear
close all
%#ok<*NOPTS>
%#ok<*ASGLU>
%#ok<*MINV>
%#ok<*AGROW>
%#ok<*FNDSB>
%#ok<*NASGU>

%% 1. Data Description
clear
clc
close all

% [1-1]. Read in the data =================================================
FILENAME = 'ai4i2020.csv';
FID = fopen(FILENAME);

if (FID < 0) % Check if the file exists
    fprintf("\nFile was not read properly.");
    fprintf("\nPlease check the file name.");
    fprintf("\nFile name should be spambase.data");
    fprintf("\nExiting the program...\n\n");
    return;
end

% Collect the data from the file
DA = csvread(FILENAME, 1, 0);
X  = DA(:,1:6);   % Feature Data X
Y_RUL = DA(:,7);  % Target  Data Y for Remaining Useful Life Analysis

fclose(FID);

% [1-2]. Plot Data ========================================================
% Separate Data into 120 Units
UIND = find(DA(:,6) == 0);
UIND = [UIND ; 10001];
UNUM = size(UIND,1)-1;
UNIT = cell(UNUM,1);
for i = 1:UNUM
    TMP = DA(UIND(i):UIND(i+1)-1,:);
    UNIT{i} = TMP;
end

% Plot Data
figure;
for j = 1:4
    subplot(2,2,j); hold on;
    for i = 1:UNUM
        scatter(UNIT{i}(:,6),UNIT{i}(:,j+1));
    end
    xlabel("Time (min)"); grid on;
    if j == 1
        title("Air Temperature vs Time");
        ylabel("Air Temperature (K)"); ylim([290 310]);
    elseif j == 2
        title("Process Temperature vs Time");
        ylabel("Process Temperature (K)"); ylim([300 320]);
    elseif j == 3
        title("Rotational Speed vs Time");
        ylabel("Rotational Speed (RPM)");
    elseif j == 4
        title("Torque vs Time");
        ylabel("Torque (Nm)");
    end 
end
sgtitle("[Data Description] Plot of Feature over Time");

% [1-3]. Plot 2D PCA ======================================================
D = 2;              % 2-Dimensional
[W,L] = PCA(X,D,0); % 0 for Non-whiten
Z = X * W;          % Please open file "PCA.m"

figure; scatter(Z(:,1),Z(:,2),'o'); grid on; 
xlabel("1st Principal Component"); ylabel("2nd Principal Component");
title("2D PCA Projection of Data");

% U_PROB = find(Y_RUL == 1);
% X_PROB = X(U_PROB,:);
% Z_PROB = X_PROB * W;
% hold on; scatter(Z_PROB(:,1),Z_PROB(:,2),'r')

%% 2. Remaining Useful Life Analysis
% [2-1]. Randomize the data ===============================================
% Seed the random number generator with zero
rng(0);
Y = Y_RUL;

% Count the number of observations and features of X
[OBSV,FEAT] = size(X);

% Randomize the data
RAN = randperm(OBSV)';
XR = X(RAN,:); % Randomize the row order of X using RAN
               % (XR : stands for X_Randomized)
YR = Y(RAN,:); % Randomize the row order of Y uisng the same RAN
               % (YR : stands for Y_Randomized)

% [2-2]. Split the data ===================================================
SPLT = ceil(OBSV*(2/3)); % 2/3 for training and 1/3 for validation

XT = XR(1:SPLT,:);       % For training (XT : stands for X_Training)
YT = YR(1:SPLT,:);       % For training (YT : stands for Y_Training)

XV = XR(SPLT+1:end,:);   % For validation (XV : stands for X_Validation)
YV = YR(SPLT+1:end,:);   % For validation (YV : stands for Y_Validation) 

% [2-3]. Z-Score the feature & validation data ============================
% Z-Score the training data
MU = mean(XT);
STD = std(XT);
XT = (XT - MU)./STD;
XV = (XV - MU)./STD; 
% Z-Score the validation data with
% respect to the training parameters
% where MU = mean(XT), STD = std(XT)

%% [2-4]. Naive Bayes Classification Model ================================
% Please open file "MultiNaiveBayes.m"
YV_Hat_NB = MultiNaiveBayes(XT,YT,XV,YV,0.0001);

% Statistical Result
ACRY_NB = sum(YV_Hat_NB == YV) / size(YV,1)
LABEL = categorical({'Long';'Medium';'Short';'Urgent'});
figure; NB = confusionchart(confusionmat(YV,YV_Hat_NB),LABEL);
NB.RowSummary = 'row-normalized'; NB.ColumnSummary='column-normalized';
title(sprintf("[Naive Bayes] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_NB));

%% [2-5]. ID3 Decision Tree Classification Model ==========================
% Decision Tree Classification Model
% Binarize the training data based on the mean of each feature
XT_BY = XT;
for i = 1:size(XT_BY,2)
    MEAN =  mean(XT_BY(:,i));
    for j = 1:size(XT_BY,1)
        if XT_BY(j,i) > MEAN
            XT_BY(j,i) = 1;
        else
            XT_BY(j,i) = 0;
        end
    end
end
XYT = [XT_BY YT];   % Binarized Training Data with Target Data

% Prepare the name of each feature
ATBT_Active = ones(1,size(XT_BY,2));
ATBT_Cell = cell(1,size(XYT,2));
for i = 1:size(XYT,2)
    ATBT_Cell{i} = sprintf('COL_%i',i);
end

% Now, Train the decision tree model using ID3
% Please Open file "MultiDecisionTree.m"
TREE = MultiDecisionTree(XYT,ATBT_Cell,ATBT_Active);
UNIQ = unique(Y)';  % Identify Unique Classes

% Next, binarize the validation data
XV_BY = XV;
for i = 1:size(XV_BY,2)
    MEAN =  mean(XV_BY(:,i));
    for j = 1:size(XV_BY,1)
        if XV_BY(j,i) > MEAN
            XV_BY(j,i) = 1;
        else
            XV_BY(j,i) = 0;
        end
    end
end
XYV = [XV_BY YV];   % Binarized Validation Data with Target Data

YV_Hat_DT = zeros(size(YV,1),1);
ATBT_Feature = ATBT_Cell(1 : length(ATBT_Cell)-1);
for k = 1 : size(YV_Hat_DT,1)
    % Please Open file "ClassifyDecisionTree.m"
    YV_Hat_DT(k,:) = ClassifyDecisionTree(TREE, ATBT_Feature, XYV(k,:),UNIQ);
end

% Statistical Result
ACRY_DT = sum(YV_Hat_DT == YV) / size(YV,1)
LABEL = categorical({'Long';'Medium';'Short';'Urgent'});
figure; DT = confusionchart(confusionmat(YV,YV_Hat_DT),LABEL);
DT.RowSummary = 'row-normalized'; DT.ColumnSummary='column-normalized';
title(sprintf("[Decision Tree] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_DT));

% If you want to see the tree structure, please run below code
% CAUTION: It generates long output
% PrintDecisionTree(TREE,'ROOT',UNIQ);

%% [2-6]. ID3 K-Nearest Neighbor Classification Model =====================
XYT = [YT XT];
XYV = [YV XV];

% Multiple K values will be explored.
% Please Open file "MultiKNN.m"
ACRY_KNN = [];
for i = 1:100:1500
    YV_Hat_KNN = MultiKNN(XYT,XYV,i);
    ACRY_KNN = [ACRY_KNN sum(YV_Hat_KNN == YV)/size(YV,1)];
end
figure; plot(1:100:1500,ACRY_KNN);
xlabel("K"); ylabel("Accuracy"); grid on;
title("Accuracy Performance of KNN with K = 1:100:1500");
% K = 601 was identified to have the highest accuracy.

% Statistical Result
YV_Hat_KNN = MultiKNN(XYT,XYV,601);
ACRY_KNN = sum(YV_Hat_KNN == YV)/size(YV,1)
figure; KNN = confusionchart(confusionmat(YV,YV_Hat_KNN),LABEL);
KNN.RowSummary = 'row-normalized'; KNN.ColumnSummary='column-normalized';
title(sprintf("[KNN w/ K = 601] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_KNN));

%% [2-7]. Logistic Regression Classification Model ========================
% Perform One vs One Multiclass Classification
XT_LRG = [ones(SPLT,1) XT];      % Add a bias feature to XT
XV_LRG = [ones(OBSV-SPLT,1) XV]; % Add a bias feature to XV

% For One vs One classification, k(k-1)/2 Classifiers are required
UNQ = unique(YT, 'rows'); % Unique Number of Classes
YH_VAL_MAT = [];          % Initialization: Predicted Solution on Validation Data
YT_MAT = [];
XT_MAT = [];

for i = 1:size(UNQ,1)   
    for j = i+1:size(UNQ,1)
        
        % Split the Y Training Data
        YT_MAT = YT(find(YT==UNQ(i)),:)-i;
        YT_MAT = [YT_MAT ; YT(find(YT==UNQ(j)),:)-(j-1)];

        % Split the X Training Data
        XT_MAT = XT_LRG(find(YT==UNQ(i)),:);
        XT_MAT = [XT_MAT ; XT_LRG(find(YT==UNQ(j)),:)];

        % Build the model based on Logistic Regression
        % Please Open file "MultiLRG.m"
        w = MultiLRG(XT_MAT, YT_MAT, i, j);
        
        % [6]. Apply the models to each validation sample
        YH_VAL = XV_LRG*w;
        
        % Apply the threshold = 0.5
        THSH = 0.5;
        for k = 1:length(YH_VAL)
            if YH_VAL(k) <= THSH  % If less than 0.5, assign i
                YH_VAL(k) = i;
            else                  % If greater than 0.5, assign j
                YH_VAL(k) = j;
            end
        end

        % Update Predicted Solution Vector
        YH_VAL_MAT = [YH_VAL_MAT , YH_VAL];
        
    end
end

% CLSV = Predicted Classes for validation data based on threshold = 0.5
YV_Hat_LRG = [];    % Class determined based on Validation Data
COUNT = [];         % Count
rng(0);             % Random Seed with 0

% Count the number of each class label row by row
for i = 1:size(UNQ,1)
    COUNT = [COUNT, sum(YH_VAL_MAT==UNQ(i),2)];
end

% Find the maximum class sum in each column
for i = 1:size(COUNT,1)
    
    MAX = max(COUNT(i,:));
    COL = find(COUNT(i,:)==MAX);
    
    % Assign Maxmimum Count Label as Predicted Label
    if (size(COL,2)==1)
        YV_Hat_LRG = [YV_Hat_LRG; UNQ(COL)];  
    % When class is a tie, assign random  
    else
        R = randperm(size(COL,2));
        YV_Hat_LRG = [YV_Hat_LRG; UNQ(R(1,1))];
    end
    
end

% Statistical Result
ACRY_LRG = sum(YV_Hat_LRG == YV) / size(YV,1)
LABEL = categorical({'Long';'Medium';'Short';'Urgent'});
figure; LRG = confusionchart(confusionmat(YV,YV_Hat_LRG),LABEL);
LRG.RowSummary = 'row-normalized'; LRG.ColumnSummary='column-normalized';
title(sprintf("[Logistic Regression] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_LRG));

%% 3. Possible Root Cause Analysis
% [3-1]. Create Balanced Dataset from Fulldata ============================
DA_PRC = DA;
rng(0);
IND = find(DA(:,8)==0);
RAN = randperm(size(IND,1))';
IND = IND(RAN);
IND = IND(1:9560);

DA_PRC(IND,:) = [];
X  = DA_PRC(:,1:7);   % Feature Data X
Y_PRC = DA_PRC(:,9);  % Target  Data Y for Possible Root Cause Analysis

% [3-2]. Randomize the data ===============================================
% Seed the random number generator with zero
rng(0);
Y = Y_PRC;

% Count the number of observations and features of X
[OBSV,FEAT] = size(X);

% Randomize the data
RAN = randperm(OBSV)';
XR = X(RAN,:); % Randomize the row order of X using RAN
               % (XR : stands for X_Randomized)
YR = Y(RAN,:); % Randomize the row order of Y uisng the same RAN
               % (YR : stands for Y_Randomized)

% [3-3]. Split the data ===================================================
SPLT = ceil(OBSV*(2/3)); % 2/3 for training and 1/3 for validation

XT = XR(1:SPLT,:);       % For training (XT : stands for X_Training)
YT = YR(1:SPLT,:);       % For training (YT : stands for Y_Training)

XV = XR(SPLT+1:end,:);   % For validation (XV : stands for X_Validation)
YV = YR(SPLT+1:end,:);   % For validation (YV : stands for Y_Validation) 

% [3-4]. Z-Score the feature & validation data ============================
% Z-Score the training data
MU = mean(XT);
STD = std(XT);
XT = (XT - MU)./STD;
XV = (XV - MU)./STD; 
% Z-Score the validation data with
% respect to the training parameters
% where MU = mean(XT), STD = std(XT)

%% [3-5]. Naive Bayes Classification Model ================================
% Please open file "MultiNaiveBayes.m"
YV_Hat_NB = MultiNaiveBayes(XT,YT,XV,YV,0.0001);

% Statistical Result
ACRY_NB = sum(YV_Hat_NB == YV) / size(YV,1)
figure; NB = confusionchart(YV,YV_Hat_NB);
NB.RowSummary = 'row-normalized'; NB.ColumnSummary='column-normalized';
title(sprintf("[Naive Bayes] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_NB));

%% [3-6]. ID3 Decision Tree Classification Model ==========================
% Decision Tree Classification Model
% Binarize the training data based on the mean of each feature
XT_BY = XT;
for i = 1:size(XT_BY,2)
    MEAN =  mean(XT_BY(:,i));
    for j = 1:size(XT_BY,1)
        if XT_BY(j,i) > MEAN
            XT_BY(j,i) = 1;
        else
            XT_BY(j,i) = 0;
        end
    end
end
XYT = [XT_BY YT];   % Binarized Training Data with Target Data

% Prepare the name of each feature
ATBT_Active = ones(1,size(XT_BY,2));
ATBT_Cell = cell(1,size(XYT,2));
for i = 1:size(XYT,2)
    ATBT_Cell{i} = sprintf('COL_%i',i);
end

% Now, Train the decision tree model using ID3
% Please Open file "MultiDecisionTree.m"
TREE = MultiDecisionTree(XYT,ATBT_Cell,ATBT_Active);
UNIQ = unique(Y)';  % Identify Unique Classes

% Next, binarize the validation data
XV_BY = XV;
for i = 1:size(XV_BY,2)
    MEAN =  mean(XV_BY(:,i));
    for j = 1:size(XV_BY,1)
        if XV_BY(j,i) > MEAN
            XV_BY(j,i) = 1;
        else
            XV_BY(j,i) = 0;
        end
    end
end
XYV = [XV_BY YV];   % Binarized Validation Data with Target Data

YV_Hat_DT = zeros(size(YV,1),1);
ATBT_Feature = ATBT_Cell(1 : length(ATBT_Cell)-1);
for k = 1 : size(YV_Hat_DT,1)
    % Please Open file "ClassifyDecisionTree.m"
    YV_Hat_DT(k,:) = ClassifyDecisionTree(TREE, ATBT_Feature, XYV(k,:),UNIQ);
end

% Statistical Result
ACRY_DT = sum(YV_Hat_DT == YV) / size(YV,1)
figure; DT = confusionchart(YV,YV_Hat_DT);
DT.RowSummary = 'row-normalized'; DT.ColumnSummary='column-normalized';
title(sprintf("[Decision Tree] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_DT));

% If you want to see the tree structure, please run below code
% CAUTION: It generates long output
% PrintDecisionTree(TREE,'ROOT',UNIQ);

%% [3-7]. ID3 K-Nearest Neighbor Classification Model =====================
XYT = [YT XT];
XYV = [YV XV];

% Multiple K values will be explored.
% Please Open file "MultiKNN.m"
ACRY_KNN = [];
for i = 1:1:20
    YV_Hat_KNN = MultiKNN(XYT,XYV,i);
    ACRY_KNN = [ACRY_KNN sum(YV_Hat_KNN == YV)/size(YV,1)];
end
figure; plot(1:1:20,ACRY_KNN);
xlabel("K"); ylabel("Accuracy"); grid on;
title("Accuracy Performance of KNN with K = 1:1:20");
% K = 14 was identified to have the highest accuracy.

% Statistical Result
YV_Hat_KNN = MultiKNN(XYT,XYV,14);
ACRY_KNN = sum(YV_Hat_KNN == YV)/size(YV,1)
figure; KNN = confusionchart(YV,YV_Hat_KNN);
KNN.RowSummary = 'row-normalized'; KNN.ColumnSummary='column-normalized';
title(sprintf("[KNN w/ K = 14] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_KNN));

%% [3-8]. Logistic Regression Classification Model ========================
% Perform One vs One Multiclass Classification
XT_LRG = [ones(SPLT,1) XT];      % Add a bias feature to XT
XV_LRG = [ones(OBSV-SPLT,1) XV]; % Add a bias feature to XV

% For One vs One classification, k(k-1)/2 Classifiers are required
UNQ = unique(YT, 'rows'); % Unique Number of Classes
YH_VAL_MAT = [];          % Initialization: Predicted Solution on Validation Data
YT_MAT = [];
XT_MAT = [];

for i = 1:size(UNQ,1)   
    for j = i+1:size(UNQ,1)
        
        % Split the Y Training Data
        YT_MAT = YT(find(YT==UNQ(i)),:)-i;
        YT_MAT = [YT_MAT ; YT(find(YT==UNQ(j)),:)-(j-1)];

        % Split the X Training Data
        XT_MAT = XT_LRG(find(YT==UNQ(i)),:);
        XT_MAT = [XT_MAT ; XT_LRG(find(YT==UNQ(j)),:)];

        % Build the model based on Logistic Regression
        % Please Open file "MultiLRG.m"
        w = MultiLRG(XT_MAT, YT_MAT, i, j);
        
        % [6]. Apply the models to each validation sample
        YH_VAL = XV_LRG*w;
        
        % Apply the threshold = 0.5
        THSH = 0.5;
        for k = 1:length(YH_VAL)
            if YH_VAL(k) <= THSH  % If less than 0.5, assign i
                YH_VAL(k) = i;
            else                  % If greater than 0.5, assign j
                YH_VAL(k) = j;
            end
        end

        % Update Predicted Solution Vector
        YH_VAL_MAT = [YH_VAL_MAT , YH_VAL];
        
    end
end

% CLSV = Predicted Classes for validation data based on threshold = 0.5
YV_Hat_LRG = [];    % Class determined based on Validation Data
COUNT = [];         % Count
rng(0);             % Random Seed with 0

% Count the number of each class label row by row
for i = 1:size(UNQ,1)
    COUNT = [COUNT, sum(YH_VAL_MAT==UNQ(i),2)];
end

% Find the maximum class sum in each column
for i = 1:size(COUNT,1)
    
    MAX = max(COUNT(i,:));
    COL = find(COUNT(i,:)==MAX);
    
    % Assign Maxmimum Count Label as Predicted Label
    if (size(COL,2)==1)
        YV_Hat_LRG = [YV_Hat_LRG; UNQ(COL)];  
    % When class is a tie, assign random  
    else
        R = randperm(size(COL,2));
        YV_Hat_LRG = [YV_Hat_LRG; UNQ(R(1,1))];
    end
    
end

% Statistical Result
ACRY_LRG = sum(YV_Hat_LRG == YV) / size(YV,1)
figure; LRG = confusionchart(YV,YV_Hat_LRG);
LRG.RowSummary = 'row-normalized'; LRG.ColumnSummary='column-normalized';
title(sprintf("[Logistic Regression] Validation Confusion Matrix (Accuracy = %.3f)",ACRY_LRG));

%% [3-9]. Logistic Regression Classification Model ========================
% Plot Data
Y = DA(:,9);
figure;
for j = 1:4
    subplot(2,2,j);
    scatter(DA(:,6),DA(:,j+1)); hold on;
    for i = 2:5
        IND_PRC = find(Y == i);
        scatter(DA(IND_PRC,6),DA(IND_PRC,j+1)); hold on;
    end
    xlabel("Time (min)"); grid on;
    if j == 1
        title("Air Temperature vs Time");
        ylabel("Air Temperature (K)"); ylim([290 310]);
        legend("Y = 1","Y = 2","Y = 3","Y = 4","Y = 5");
    elseif j == 2
        title("Process Temperature vs Time");
        ylabel("Process Temperature (K)"); ylim([300 320]);
        legend("Y = 1","Y = 2","Y = 3","Y = 4","Y = 5");
    elseif j == 3
        title("Rotational Speed vs Time");
        ylabel("Rotational Speed (RPM)");
        legend("Y = 1","Y = 2","Y = 3","Y = 4","Y = 5");
    elseif j == 4
        title("Torque vs Time");
        ylabel("Torque (Nm)");
        legend("Y = 1","Y = 2","Y = 3","Y = 4","Y = 5");
    end 
end
sgtitle("Failure Distribution over Full Data");
