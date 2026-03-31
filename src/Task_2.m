%% Hard margin with linear kernel
g_Tr = wo'*stndzd_Tr+bo;

g_Te = wo'*stndzd_Te+bo;
% Testing and training accuracies
Acc_Tr = 0;
pred_Tr = sign(g_Tr);
for i = 1:2000   
    
    if pred_Tr(i) == TrLabel(i,1)
        Acc_Tr = Acc_Tr+1;
    end
end
Training_Accuracy = Acc_Tr*100/2000

Acc_Te = 0;
pred_Te = sign(g_Te);
for i = 1:1536    
    
    if pred_Te(i) == TeLabel(i,1)
        Acc_Te = Acc_Te+1;
    end
end
Testing_Accuracy = Acc_Te*100/1536

%% Hard/Soft margin with polynomial kernel
wo_Tr = sum(x.*TrLabel.*(stndzd_Tr'*stndzd_Tr+1).^p,1)';
bo = mean(TrLabel(idx)-wo_Tr(idx));
g_Tr = wo_Tr + bo;

wo_Te = sum(x.*TrLabel.*(stndzd_Tr'*stndzd_Te+1).^p,1)';
g_Te = wo_Te + bo;

% Testing and training accuracies
Acc_Tr = 0;
pred_Tr = sign(g_Tr);
for i = 1:2000   
    
    if pred_Tr(i) == TrLabel(i,1)
        Acc_Tr = Acc_Tr+1;
    end
end
Training_Accuracy = Acc_Tr*100/2000

Acc_Te = 0;
pred_Te = sign(g_Te);
for i = 1:1536    
    
    if pred_Te(i) == TeLabel(i,1)
        Acc_Te = Acc_Te+1;
    end
end
Testing_Accuracy = Acc_Te*100/1536