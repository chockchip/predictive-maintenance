function [YV_Hat] = ClassifyDecisionTree(TREE,ATBT_Feature,XYV,UNIQ)
    % June Kwon
	%#ok<*FNDSB>
    % Assign the corresponding value at the leaf
    for i = UNIQ
       if (strcmp(TREE.VALUE, num2str(i)))
           YV_Hat = i;
           return
       end
    end
    
    % If current node is labeled as an attribute,
    % follow the correct branch by looking up index in attributes, and recur
    IDNX = find(ismember(ATBT_Feature,TREE.VALUE) == 1);
    if (XYV(1,IDNX))  % If attribute is true for this instance
        % Recur down the right side
        YV_Hat = ClassifyDecisionTree(TREE.RIGHT,ATBT_Feature,XYV,UNIQ); 
    else
        % Recur down the left side
        YV_Hat = ClassifyDecisionTree(TREE.LEFT,ATBT_Feature,XYV,UNIQ);
    end
    
    return
end