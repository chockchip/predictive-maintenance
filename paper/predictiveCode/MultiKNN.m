function YV_HAT = MultiKNN(XYT,XYV,K)

% Find distance with all training datapoints
% Then Sort & Poll
% June Kwon
%#ok<*AGROW>
%#ok<*HISTC> 

DIST = zeros(size(XYT,1),1);
YV_PRED = [];
    for i = 1 : size(XYV)
        COL = XYV(i,:); % Column to compare

        % Perform Squared Euclidean Distance
        for j = 2 : size(XYT,2)
            DIST(:,1) =  DIST(:,1) + (XYT(:,j)-COL(j)).^2;
        end
        DIST(:,1) = sqrt(DIST(:,1));
        
        YT = XYT(:,1);
        DIST(:,2) = YT;
        POLL = sortrows(DIST,1);

        % [Tiebreak] In case K is even
        % If Tiebreak happens, class with minimum sum of distances
        % is yielded to the predicted validation data.
        if (mod(K,2) == 1) % If K is odd number
            YV_PRED(i) = mode(POLL(1:K,2)); 
        else % If K is even number
            TMP  = POLL(1:K,2); % Temporary Storage
            UNIQ = unique(TMP); % Unique Classes
            CLSN = size(UNIQ);  % Number of Classes
            BINC = histc(TMP,UNIQ); % Count number of bins
            MX = max(BINC);     % Highest Frequency
            % if number of class == 2 && highest
            % frequency is K/2, then there is tie
            TIE = (CLSN == 2) & (MX == K/2);
            % Yield the class which is at the closest distance
            YV_PRED(i) = mode(POLL(1:K-TIE,2));    
        end
    end
YV_HAT = YV_PRED';

end
