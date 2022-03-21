function [] = PrintDecisionTree(TREE,PARENT,UNIQ)
    % June Kwon
	
    for i = UNIQ
       if (strcmp(TREE.VALUE, num2str(i)))
           fprintf('PARENT: %s\t%i\n', PARENT,i);
           return
       end
    end
    
    fprintf('PARENT: %s\tATTRIBUTE: %s\tFalseChild %s\tTrueChild: %s\n', ...
        PARENT,TREE.VALUE,TREE.LEFT.VALUE,TREE.RIGHT.VALUE);

    % Recur the LEFT subtree
    PrintDecisionTree(TREE.LEFT,TREE.VALUE,UNIQ);

    % Recur the RIGHT subtree
    PrintDecisionTree(TREE.RIGHT,TREE.VALUE,UNIQ);

end